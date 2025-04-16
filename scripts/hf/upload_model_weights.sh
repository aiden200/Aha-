#!/bin/bash

huggingface-cli login
# Your model repo ID
REPO_ID="aiden200/aha"
FILE_DIR="outputs/aha/checkpoint-350/"


# Files to upload
FILES=(
  "adapter_config.json"
  "adapter_model.safetensors"
  "added_tokens.json"
  "merges.txt"
  "special_tokens_map.json"
  "tokenizer.json"
  "vocab.json"
)


# Upload each file one by one
for file in "${FILES[@]}"; do
  FILE_PATH="${FILE_DIR}${file}"
  if [ -f "$FILE_PATH" ]; then
    echo "📤 Uploading $file to $REPO_ID..."
    huggingface-cli upload "$REPO_ID" "$FILE_PATH" "$file"
  else
    echo "❌ File not found: $FILE_PATH"
  fi
done