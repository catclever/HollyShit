#!/bin/bash
# Cleans up wandb local cache older than 7 days
find wandb/ -type d -name "run-*" -mtime +7 -exec rm -rf {} +
find wandb/ -type d -name "offline-run-*" -mtime +7 -exec rm -rf {} +
echo "Old wandb local logs cleaned up."
