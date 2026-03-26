import os
import json
import signal
import sys
import argparse
from typing import Dict, Any

class Checkpointer:
    """
    A unified checkpointer that manages saving/loading Model weights and Dataloader states.
    It automatically catches SIGINT/SIGTERM to perform emergency saves before exiting.
    """
    def __init__(self, out_dir: str, prefix: str = None):
        self.out_dir = out_dir
        
        if prefix == "TIMESTAMP":
            from datetime import datetime
            self.prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.prefix = "" if prefix is None else prefix
            
        os.makedirs(self.out_dir, exist_ok=True)
        
        self._models = {}
        self._dataloaders = {}
        self._args = None
        self.current_step = 0

        # Register robust interrupt traps globally
        self._setup_signals()

    def register_model(self, name: str, model):
        """Register an MLX nn.Module to be saved/loaded as .safetensors"""
        self._models[name] = model
        
    def register_dataloader(self, name: str, dataloader):
        """Register a custom dataloader that implements state_dict/load_state_dict"""
        self._dataloaders[name] = dataloader
        
    def register_args(self, args: argparse.Namespace):
        """Register parsed CLI arguments to be saved inside the checkpoint."""
        self._args = args

    def _setup_signals(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        # Translate UNIX signals into Python's KeyboardInterrupt
        raise KeyboardInterrupt

    def save(self, step: int, is_emergency: bool = False):
        """Saves registered models and dataloaders."""
        base_name = "latest_emergency" if is_emergency else f"step_{step}"
        folder_name = f"{self.prefix}_{base_name}" if self.prefix else base_name
        save_path = os.path.join(self.out_dir, folder_name)
        
        if is_emergency:
            print(f"\n[Interrupt] Caught kill signal at step {step}! Saving emergency checkpoint to {save_path}...")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save models
        for name, model in self._models.items():
            if hasattr(model, "save_weights"):
                model.save_weights(f"{save_path}/{name}.safetensors")
            else:
                try:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        torch.save(model.state_dict(), f"{save_path}/{name}.pt")
                    else:
                        print(f"Warning: Unrecognized model type for {name}")
                except ImportError:
                    print(f"Warning: Model {name} looks like PyTorch but torch is not installed.")
            
        # Save dataloaders
        for name, loader in self._dataloaders.items():
            if hasattr(loader, "state_dict"):
                state = loader.state_dict()
                state["global_step"] = step
                with open(f"{save_path}/{name}.json", "w") as f:
                    json.dump(state, f)
                    
        # Save training args
        if self._args is not None:
            # We must convert argparse.Namespace to dict 
            with open(f"{save_path}/training_args.json", "w") as f:
                json.dump(vars(self._args), f)
                
        if is_emergency:
            print(f"[Checkpoint] Emergency save completed at {save_path}. Safely exiting process.")
            sys.exit(0)
        else:
            print(f"[Checkpoint] Saved at step {step} to {save_path}")
            
    def load_latest(self) -> int:
        """Automatically finds the checkpoint with the highest global_step."""
        if not os.path.exists(self.out_dir):
            print(f"No existing checkpoints found in {self.out_dir}. Starting fresh.")
            return 0
            
        latest_dir = None
        max_step = -1
        
        for d in os.listdir(self.out_dir):
            d_path = os.path.join(self.out_dir, d)
            if not os.path.isdir(d_path):
                continue
                
            # Filter strictly by prefix if provided
            if self.prefix and self.prefix != "TIMESTAMP":
                if not d.startswith(self.prefix + "_"):
                    continue
                    
            if d.endswith("latest_emergency") or "_step_" in d or d.startswith("step_"):
                # Always attempt to extract exact step from dataloader state first
                step = -1
                for file in os.listdir(d_path):
                    if file.endswith(".json") and "dataloader" in file:
                        try:
                            with open(os.path.join(d_path, file), "r") as f:
                                state = json.load(f)
                                step = state.get("global_step", -1)
                        except:
                            pass
                            
                # Fallback to folder name parsing if JSON parsing failed
                if step == -1 and "_step_" in d:
                    parts = d.split("_step_")
                    if len(parts) == 2 and parts[1].isdigit():
                        step = int(parts[1])
                        
                if step > max_step:
                    max_step = step
                    latest_dir = d
                    
        if not latest_dir:
            print(f"No valid checkpoints found in {self.out_dir}. Starting fresh.")
            return 0
            
        # Extract the prefix to continue seamlessly
        if latest_dir.endswith("latest_emergency"):
            if latest_dir == "latest_emergency":
                self.prefix = ""
            else:
                self.prefix = latest_dir.replace("_latest_emergency", "")
        else:
            if latest_dir.startswith("step_"):
                self.prefix = ""
            else:
                self.prefix = latest_dir.split("_step_")[0]
                
        print(f"Auto-resume overriding future saves to use prefix: '{self.prefix}' (Extracted from {latest_dir})")
        load_path = os.path.join(self.out_dir, latest_dir)
        return self.load(load_path)
            
    def load(self, load_path: str) -> int:
        """Loads registered models and dataloaders, returning the resumed global_step."""
        print(f"Resuming from checkpoint {load_path}...")
        step = 0
        
        for name, model in self._models.items():
            try:
                if hasattr(model, "load_weights"):
                    target_file = f"{load_path}/{name}.safetensors"
                    # Compatibility specific for the adapter to fuser rename
                    if name == "sense_fuser" and not os.path.exists(target_file):
                        fallback_file = f"{load_path}/sense_adapter.safetensors"
                        if os.path.exists(fallback_file):
                            target_file = fallback_file
                            print(f"Info: Loaded legacy '{fallback_file}' for model '{name}'.")
                    model.load_weights(target_file)
                else:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        target_file = f"{load_path}/{name}.pt"
                        model.load_state_dict(torch.load(target_file, weights_only=True))
            except Exception as e:
                print(f"Warning: Could not load weights for {name}: {e}")
                
        for name, loader in self._dataloaders.items():
            if hasattr(loader, "load_state_dict"):
                try:
                    with open(f"{load_path}/{name}.json", "r") as f:
                        state = json.load(f)
                        loader.load_state_dict(state)
                        # Extract the global step tracked within the dataloader state
                        step = state.get("global_step", 0)
                except Exception as e:
                    print(f"Warning: Could not load state for dataloader {name}: {e}")
                    
        self.current_step = step
        return step
