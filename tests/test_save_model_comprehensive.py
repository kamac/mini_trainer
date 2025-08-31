"""
Comprehensive tests for save_model function.

Tests validate:
1. Model saves as expected on the main process
2. Model casts params to save_dtype as expected  
3. State dict is offloaded onto the CPU as expected
4. Non-main processes do not materialize the state dict
5. prepare_state_dict_for_save happens correctly for OSFT models on main process only
"""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call, ANY
import pytest

import torch
import torch.nn as nn
import torch.distributed as dist

from mini_trainer.train import save_model


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 5)
        self.config = MagicMock()
        self.config.torch_dtype = torch.bfloat16
        
    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


class SimpleOSFTModel(SimpleModel):
    """Simple OSFT model with prepare_state_dict_for_save method."""
    def __init__(self):
        super().__init__()
        self.name_mapping = {"layer1.weight": "layer1_weight_safe"}
        
    def prepare_state_dict_for_save(self, state_dict):
        """Mock OSFT preparation - just adds a marker to verify it was called."""
        # Add a marker to indicate this was called
        new_state_dict = {}
        for k, v in state_dict.items():
            # Add "_osft_prepared" suffix to keys to verify this was called
            new_state_dict[k + "_osft_prepared"] = v
        return new_state_dict


class TestSaveModelComprehensive:
    """Comprehensive test suite for save_model function."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model with real tensors."""
        model = SimpleModel()
        # Set weights to known values for testing
        with torch.no_grad():
            model.layer1.weight.fill_(1.0)
            model.layer2.weight.fill_(2.0)
        return model
    
    @pytest.fixture
    def simple_osft_model(self):
        """Create a simple OSFT model."""
        model = SimpleOSFTModel()
        with torch.no_grad():
            model.layer1.weight.fill_(1.0)
            model.layer2.weight.fill_(2.0)
        return model
    
    @patch.dict(os.environ, {'RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    @patch('mini_trainer.train.log_rank_0')
    def test_save_model_main_process_saves(self, mock_log, mock_save_file, mock_split,
                                          mock_tokenizer, mock_barrier, mock_rank, simple_model):
        """Test that model saves correctly on the main process."""
        # Wrap model to simulate FSDP
        fsdp_model = MagicMock()
        fsdp_model.module = simple_model
        
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock split to not shard
        mock_split_result = MagicMock()
        mock_split_result.filename_to_tensors = {'model.safetensors': ['layer1.weight', 'layer2.weight']}
        mock_split_result.is_sharded = False
        mock_split.return_value = mock_split_result
        
        # Mock get_model_state_dict to return actual state dict
        with patch('torch.distributed.checkpoint.state_dict.get_model_state_dict') as mock_get_state:
            # Return actual model state dict
            actual_state_dict = {
                'layer1.weight': simple_model.layer1.weight.clone(),
                'layer1.bias': simple_model.layer1.bias.clone(),
                'layer2.weight': simple_model.layer2.weight.clone(),
                'layer2.bias': simple_model.layer2.bias.clone(),
            }
            mock_get_state.return_value = actual_state_dict
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_model(
                    fsdp_model,
                    samples_seen=1000,
                    output_dir=temp_dir,
                    model_name_or_path="test/model"
                )
            
            # Verify state dict was retrieved with correct options
            mock_get_state.assert_called_once_with(
                fsdp_model,
                options=ANY
            )
            
            # Check the options passed
            call_args = mock_get_state.call_args
            options = call_args.kwargs['options']
            assert options.full_state_dict == True
            assert options.cpu_offload == True
            assert options.broadcast_from_rank0 == False
            
            # Verify save was called (main process)
            mock_save_file.assert_called()
            
            # Verify config and tokenizer were saved
            simple_model.config.to_json_file.assert_called_once()
            mock_tokenizer_instance.save_pretrained.assert_called_once()
    
    @patch.dict(os.environ, {'RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    @patch('mini_trainer.train.log_rank_0')
    def test_save_model_dtype_casting(self, mock_log, mock_save_file, mock_split,
                                     mock_tokenizer, mock_barrier, mock_rank, simple_model):
        """Test that model parameters are cast to save_dtype as expected."""
        # Set model dtype to bfloat16
        simple_model.config.torch_dtype = torch.bfloat16
        
        # Create FP32 tensors (simulating training in FP32)
        fp32_state_dict = {
            'layer1.weight': torch.randn(10, 10, dtype=torch.float32),
            'layer1.bias': torch.randn(10, dtype=torch.float32),
            'layer2.weight': torch.randn(5, 10, dtype=torch.float32),
            'layer2.bias': torch.randn(5, dtype=torch.float32),
        }
        
        fsdp_model = MagicMock()
        fsdp_model.module = simple_model
        
        mock_tokenizer.return_value = MagicMock()
        
        mock_split_result = MagicMock()
        mock_split_result.filename_to_tensors = {'model.safetensors': list(fp32_state_dict.keys())}
        mock_split_result.is_sharded = False
        mock_split.return_value = mock_split_result
        
        with patch('torch.distributed.checkpoint.state_dict.get_model_state_dict') as mock_get_state:
            mock_get_state.return_value = fp32_state_dict.copy()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_model(
                    fsdp_model,
                    samples_seen=1000,
                    output_dir=temp_dir,
                    model_name_or_path="test/model"
                )
            
            # Verify save_file was called with bfloat16 tensors
            saved_tensors = mock_save_file.call_args[0][0]
            for key, tensor in saved_tensors.items():
                assert tensor.dtype == torch.bfloat16, f"Tensor {key} not cast to bfloat16"
                assert tensor.device == torch.device('cpu'), f"Tensor {key} not on CPU"
    
    @patch.dict(os.environ, {'RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('mini_trainer.train.log_rank_0')
    def test_save_model_cpu_offload(self, mock_log, mock_tokenizer, mock_barrier, mock_rank, simple_model):
        """Test that state dict is offloaded to CPU as expected."""
        fsdp_model = MagicMock()
        fsdp_model.module = simple_model
        
        # Create test tensors - some on GPU if available, otherwise CPU
        if torch.cuda.is_available():
            test_tensors = {
                'layer1.weight': torch.randn(10, 10, device='cuda'),
                'layer2.weight': torch.randn(5, 10, device='cuda'),
            }
        else:
            test_tensors = {
                'layer1.weight': torch.randn(10, 10),
                'layer2.weight': torch.randn(5, 10),
            }
        
        mock_tokenizer.return_value = MagicMock()
        
        saved_files = []
        
        # Mock the functions we need
        with patch('torch.distributed.checkpoint.state_dict.get_model_state_dict') as mock_get_state, \
             patch('huggingface_hub.split_torch_state_dict_into_shards') as mock_split, \
             patch('safetensors.torch.save_file') as mock_save_file:
            
            # get_model_state_dict returns our test tensors
            mock_get_state.return_value = test_tensors.copy()
            
            # split_torch_state_dict_into_shards just returns metadata about sharding
            mock_split.return_value = MagicMock(
                filename_to_tensors={'model.safetensors': list(test_tensors.keys())},
                is_sharded=False
            )
            
            # Capture what gets saved
            def save_file_side_effect(tensors_dict, path):
                saved_files.append((tensors_dict, path))
            
            mock_save_file.side_effect = save_file_side_effect
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_model(
                    fsdp_model,
                    samples_seen=1000,
                    output_dir=temp_dir,
                    model_name_or_path="test/model"
                )
            
            # Verify get_model_state_dict was called with cpu_offload=True
            call_args = mock_get_state.call_args
            assert call_args is not None
            kwargs = call_args[1] if len(call_args) > 1 else {}
            if 'options' in kwargs:
                assert kwargs['options'].cpu_offload == True
            
            # Verify tensors were saved and are on CPU with correct dtype
            assert len(saved_files) > 0, "No files were saved"
            saved_tensors, _ = saved_files[0]
            
            for key in test_tensors.keys():
                assert key in saved_tensors
                tensor = saved_tensors[key]
                # Verify tensor is on CPU
                assert tensor.device.type == 'cpu', f"Tensor {key} not on CPU"
                # Verify dtype matches save_dtype
                assert tensor.dtype == simple_model.config.torch_dtype, \
                    f"Tensor {key} has wrong dtype"
    
    @patch.dict(os.environ, {'RANK': '1', 'LOCAL_WORLD_SIZE': '2', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=1)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    @patch('mini_trainer.train.log_rank_0')
    def test_non_main_process_no_materialize(self, mock_log, mock_save_file, mock_split,
                                            mock_tokenizer, mock_barrier, mock_get_state,
                                            mock_rank, simple_model):
        """Test that non-main processes do not materialize the full state dict."""
        fsdp_model = MagicMock()
        fsdp_model.module = simple_model
        
        # For non-main process, get_model_state_dict should return empty or minimal dict
        # since broadcast_from_rank0=False
        mock_get_state.return_value = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Verify get_model_state_dict was called with broadcast_from_rank0=False
        call_args = mock_get_state.call_args
        if call_args[1]:  # If there are kwargs
            options = call_args[1].get('options')
            if options:
                assert options.broadcast_from_rank0 == False
        
        # Verify non-main process doesn't save
        mock_save_file.assert_not_called()
        mock_split.assert_not_called()
        
        # Verify barrier was called for synchronization
        mock_barrier.assert_called()
    
    @patch.dict(os.environ, {'RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    @patch('mini_trainer.train.log_rank_0')
    def test_osft_model_prepare_main_process(self, mock_log, mock_save_file, mock_split,
                                            mock_tokenizer, mock_barrier, mock_get_state,
                                            mock_rank, simple_osft_model):
        """Test that OSFT model's prepare_state_dict_for_save is called on main process."""
        fsdp_model = MagicMock()
        fsdp_model.module = simple_osft_model
        
        original_state_dict = {
            'layer1.weight': torch.randn(10, 10),
            'layer2.weight': torch.randn(5, 10),
        }
        mock_get_state.return_value = original_state_dict.copy()
        
        mock_tokenizer.return_value = MagicMock()
        
        # We need to set up split to use the actual transformed keys
        def split_side_effect(state_dict, **kwargs):
            # Get the actual keys from the transformed state dict
            actual_keys = list(state_dict.keys())
            return MagicMock(
                filename_to_tensors={'model.safetensors': actual_keys},
                is_sharded=False
            )
        
        mock_split.side_effect = split_side_effect
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Verify prepare_state_dict_for_save was called (via the key modification)
        saved_tensors = mock_save_file.call_args[0][0]
        # Keys should have "_osft_prepared" suffix from our mock prepare method
        assert all('_osft_prepared' in key for key in saved_tensors.keys()), \
            "prepare_state_dict_for_save was not called or didn't modify keys"
    
    @patch.dict(os.environ, {'RANK': '1', 'LOCAL_WORLD_SIZE': '2', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=1)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    @patch('mini_trainer.train.log_rank_0')
    def test_osft_model_prepare_non_main_process(self, mock_log, mock_get_state,
                                                mock_barrier, mock_rank, simple_osft_model):
        """Test that OSFT prepare_state_dict_for_save is NOT called on non-main process."""
        fsdp_model = MagicMock()
        fsdp_model.module = simple_osft_model
        
        # Spy on prepare_state_dict_for_save
        original_prepare = simple_osft_model.prepare_state_dict_for_save
        simple_osft_model.prepare_state_dict_for_save = MagicMock(side_effect=original_prepare)
        
        mock_get_state.return_value = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Verify prepare_state_dict_for_save was NOT called on non-main process
        simple_osft_model.prepare_state_dict_for_save.assert_not_called()
    
    @patch.dict(os.environ, {'RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'NODE_RANK': '1'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('torch.distributed.checkpoint.state_dict.get_model_state_dict')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('huggingface_hub.split_torch_state_dict_into_shards')
    @patch('safetensors.torch.save_file')
    def test_osft_non_main_node_behavior(self, mock_save_file, mock_split,
                                       mock_tokenizer, mock_get_state,
                                       mock_barrier, mock_rank, simple_osft_model):
        """Test that non-main nodes behave correctly during OSFT checkpoint preparation."""
        fsdp_model = MagicMock()
        fsdp_model.module = simple_osft_model
        
        mock_get_state.return_value = {}
        mock_tokenizer.return_value = MagicMock()
        mock_split.return_value = MagicMock(
            filename_to_tensors={'model.safetensors': []},
            is_sharded=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_model(
                fsdp_model,
                samples_seen=1000,
                output_dir=temp_dir,
                model_name_or_path="test/model"
            )
        
        # Verify barrier was called for synchronization
        mock_barrier.assert_called()
        # Since NODE_RANK=1 (not main node) but RANK=0 (main process on this node),
        # it should still save but would have logged a message (which we're not testing)
    
    @patch.dict(os.environ, {'RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'NODE_RANK': '0'})
    @patch('mini_trainer.train.torch.distributed.get_rank', return_value=0)
    @patch('mini_trainer.train.torch.distributed.barrier')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('mini_trainer.train.log_rank_0')
    def test_save_model_dtype_conversion_mixed(self, mock_log, mock_tokenizer, mock_barrier,
                                          mock_rank, simple_model):
        """Test that mixed dtype tensors are cast correctly to save_dtype."""
        simple_model.config.torch_dtype = torch.bfloat16
        
        # Create mixed dtype state dict
        mixed_state_dict = {
            'layer1.weight': torch.randn(10, 10, dtype=torch.float32),
            'layer1.bias': torch.randn(10, dtype=torch.float32),
            'layer2.weight': torch.randn(5, 10, dtype=torch.float16),  # Different dtype
            'layer2.bias': torch.randn(5, dtype=torch.bfloat16),  # Already correct dtype
        }
        
        fsdp_model = MagicMock()
        fsdp_model.module = simple_model
        
        mock_tokenizer.return_value = MagicMock()
        
        # Store original dtypes for verification
        original_dtypes = {k: v.dtype for k, v in mixed_state_dict.items()}
        
        saved_files = []
        
        with patch('torch.distributed.checkpoint.state_dict.get_model_state_dict') as mock_get_state, \
             patch('huggingface_hub.split_torch_state_dict_into_shards') as mock_split, \
             patch('safetensors.torch.save_file') as mock_save_file:
            
            mock_get_state.return_value = mixed_state_dict.copy()
            mock_split.return_value = MagicMock(
                filename_to_tensors={'model.safetensors': list(mixed_state_dict.keys())},
                is_sharded=False
            )
            
            # Capture what gets saved
            def save_file_side_effect(tensors_dict, path):
                saved_files.append((tensors_dict, path))
            
            mock_save_file.side_effect = save_file_side_effect
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_model(
                    fsdp_model,
                    samples_seen=1000,
                    output_dir=temp_dir,
                    model_name_or_path="test/model"
                )
            
            # Verify tensors were saved
            assert len(saved_files) > 0, "No files were saved"
            saved_dict, _ = saved_files[0]
            
            # Verify all tensors were processed and converted to bfloat16
            for key in mixed_state_dict.keys():
                assert key in saved_dict, f"Key {key} not in saved dict"
                
                saved_tensor = saved_dict[key]
                # All tensors should be converted to bfloat16 (the save_dtype)
                assert saved_tensor.dtype == torch.bfloat16, \
                    f"Tensor {key} has dtype {saved_tensor.dtype}, expected bfloat16 (was {original_dtypes[key]})"
                # Also check they're on CPU
                assert saved_tensor.device.type == 'cpu', \
                    f"Tensor {key} not on CPU"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

