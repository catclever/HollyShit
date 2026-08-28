import os
import shutil
import time
from datetime import datetime, timedelta
import glob

try:
    import wandb
except ImportError:
    print("Please install wandb: pip install wandb")
    exit(1)

# Configuration
WANDB_DIR = "wandb"
PROJECT_NAME = "HollyShit"  # Adjust if your wandb project name is different
JUNK_DURATION_SECS = 120    # Runs shorter than 2 mins are considered junk
JUNK_STEPS = 100            # Runs with fewer than 100 steps are considered junk
REMOTE_EXPIRE_DAYS = 60     # Remote runs older than 60 days are deleted

def get_run_id_from_path(path):
    basename = os.path.basename(path)
    if "-" in basename:
        return basename.split("-")[-1]
    return None

def clean_wandb():
    api = None
    try:
        api = wandb.Api()
        # Ping to verify auth
        api.default_entity
        print("Successfully connected to WandB server.")
    except Exception as e:
        print("Not logged into WandB server yet. Will only perform local offline cleanup.")
        api = None
    
    print("\n--- 1 & 2. Scanning Local Runs & Identifying Junk ---")
    if os.path.exists(WANDB_DIR):
        local_runs = glob.glob(os.path.join(WANDB_DIR, "*run-*"))
        for local_path in local_runs:
            run_id = get_run_id_from_path(local_path)
            if not run_id:
                continue
                
            print(f"Checking local run: {run_id}")
            
            is_synced = False
            remote_run = None
            if api:
                # Check if it exists remotely
                try:
                    remote_run = api.run(f"{PROJECT_NAME}/{run_id}")
                    is_synced = True
                except:
                    pass

            # Determine if it's junk
            is_junk = False
            if remote_run:
                duration = remote_run.summary.get('_runtime', 0)
                steps = remote_run.summary.get('_step', 0)
                if duration < JUNK_DURATION_SECS or steps < JUNK_STEPS:
                    is_junk = True
            else:
                # Offline junk estimation
                size_bytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(local_path) for f in fns)
                if size_bytes < 1024 * 1024 * 2: # < 2MB is definitely a crash/junk
                    is_junk = True
            
            # Action 1: Delete junk (local and remote)
            if is_junk:
                print(f"  -> Marked as JUNK (<2MB offline or short remote). Deleting locally and remotely...")
                if remote_run:
                    try:
                        remote_run.delete()
                    except: pass
                shutil.rmtree(local_path, ignore_errors=True)
                continue
                
            # Action 2: Delete local if synced
            if is_synced:
                print(f"  -> Run is safely synced to server. Deleting local cache...")
                shutil.rmtree(local_path, ignore_errors=True)
            else:
                print(f"  -> Run is offline/unsynced. Keeping local cache.")
    else:
        print("No local wandb directory found.")
        
    if api:
        print("\n--- 3. Cleaning Remote Runs Older Than 60 Days ---")
        try:
            runs = api.runs(PROJECT_NAME)
            cutoff_date = datetime.now() - timedelta(days=REMOTE_EXPIRE_DAYS)
            for run in runs:
                try:
                    updated_at = datetime.strptime(run.updated_at, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    updated_at = datetime.now()
                    
                if updated_at < cutoff_date:
                    print(f"Deleting remote run {run.id} (Last updated: {updated_at})...")
                    run.delete()
            print("Remote cleanup finished.")
        except Exception as e:
            print(f"Could not scan remote runs for project {PROJECT_NAME}: {e}")

if __name__ == '__main__':
    clean_wandb()
