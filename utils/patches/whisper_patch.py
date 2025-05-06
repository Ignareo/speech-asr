"""
Whisper patch module for handling the "batch decoding is not implemented" issue in FunASR.

This module monkey-patches the WhisperWarp class in FunASR to handle batch processing
by processing one item at a time instead of throwing an error.
"""

import logging
import time
import sys
import os
import torch
import importlib
import importlib.util
import types
import numpy as np
from typing import List, Dict, Any, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Try to import FunASR's WhisperWarp class
try:
    from funasr.models.whisper.model import WhisperWarp
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
    import whisper
    HAS_FUNASR = True
    logging.info("Found FunASR and Whisper - Patch can be applied")
except ImportError as e:
    HAS_FUNASR = False
    logging.warning(f"FunASR or Whisper not found, WhisperPatch will not be available. Error: {e}")

# Original inference method to save
original_inference = None

def patched_inference(
    self,
    data_in,
    data_lengths=None,
    key: list = None,
    tokenizer=None,
    frontend=None,
    **kwargs,
):
    """
    Patched version of WhisperWarp.inference that handles batch processing.
    Instead of raising an error for batch_size > 1, it processes items one by one.
    """
    # Import tables in the function to avoid circular imports
    from funasr.register import tables
    
    # Get batch size and cap it to a reasonable value to prevent excessive processing time
    batch_size = kwargs.get("batch_size", 1)
    # If batch_size is extremely large (from batch_size_s parameter in seconds), cap it
    if batch_size > 100:  # This is arbitrary but helps prevent very slow processing
        logging.warning(f"Batch size {batch_size} is too large, capping to 5 for better performance")
        kwargs["batch_size"] = 5
        batch_size = 5
    
    logging.info(f"WhisperWarp patched inference called with batch_size={batch_size}")
    
    if batch_size > 1:
        # Process each sample individually and combine results
        logging.info(f"Handling batch of size {batch_size} by processing one by one")
        results = []
        meta_data = {}
        
        # Check if data_in is a list/batch
        if isinstance(data_in, list) or (isinstance(data_in, torch.Tensor) and data_in.dim() > 2):
            for i in range(len(data_in)):
                logging.info(f"Processing item {i+1}/{len(data_in)} in batch")
                single_data = data_in[i]
                single_length = data_lengths[i] if data_lengths is not None else None
                single_key = [key[i]] if key is not None else None
                
                # Set batch_size to 1 for processing single sample
                kwargs_copy = kwargs.copy()
                kwargs_copy["batch_size"] = 1
                
                try:
                    # Process single sample
                    single_result, single_meta = original_inference(
                        self, 
                        single_data, 
                        single_length, 
                        single_key, 
                        tokenizer, 
                        frontend, 
                        **kwargs_copy
                    )
                    results.extend(single_result)
                    
                    # Combine meta data (simple version)
                    if i == 0:
                        meta_data = single_meta
                except Exception as e:
                    logging.error(f"Error processing batch item {i}: {str(e)}")
                    raise
            
            logging.info(f"Successfully processed all {len(data_in)} items in batch")
            return results, meta_data
        else:
            logging.info(f"Data is not a list or batch-able tensor. Setting batch_size=1")
            # If not a proper batch that we can iterate, fall back to original behavior
            # but with batch_size=1 to avoid the error
            kwargs["batch_size"] = 1
    
    # Use the original method with batch_size=1
    logging.info("Calling original inference method with batch_size=1")
    return original_inference(self, data_in, data_lengths, key, tokenizer, frontend, **kwargs)

def apply_whisper_patch():
    """
    Apply the patch to the WhisperWarp class to handle batch processing.
    This should be called before using the Whisper model in FunASR.
    """
    global original_inference
    
    if not HAS_FUNASR:
        logging.warning("FunASR not found, cannot apply Whisper patch")
        return False
    
    # Save the original method and patch it
    if original_inference is None:  # Only patch once
        logging.info("Applying patch to WhisperWarp.inference...")
        original_inference = WhisperWarp.inference
        WhisperWarp.inference = patched_inference
        
        # Also try to patch AutoModel if available
        try:
            from funasr.auto.auto_model import AutoModel
            
            # Check if model is a Whisper model
            original_generate = AutoModel.generate
            
            def patched_generate(self, input, input_len=None, **cfg):
                """
                Patched version of generate that adjusts batch_size_s for Whisper models.
                """
                # Check if the model is a WhisperWarp model
                if hasattr(self, 'model') and isinstance(self.model, WhisperWarp):
                    if 'batch_size_s' in cfg and cfg['batch_size_s'] > 5:
                        logging.warning(f"Reducing batch_size_s from {cfg['batch_size_s']} to 5 for Whisper model")
                        cfg['batch_size_s'] = 5
                
                # Call the original generate method
                return original_generate(self, input, input_len, **cfg)
            
            # Apply the patch
            AutoModel.generate = patched_generate
            logging.info("Successfully patched AutoModel.generate to reduce batch size for Whisper models")
        except Exception as e:
            logging.warning(f"Could not patch AutoModel: {e}")
        
        logging.info("Successfully patched WhisperWarp.inference to handle batch processing")
        return True
    else:
        logging.info("Patch already applied to WhisperWarp.inference")
    
    return False

# Automatically apply the patch when this module is imported
if HAS_FUNASR:
    apply_whisper_patch()
    logging.info("Whisper patch has been applied and is ready for use") 