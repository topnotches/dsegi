#!/bin/bash

# Set the dataset directory
DATASET_DIR="dataset"
MNIST_URL_BASE="http://yann.lecun.com/exdb/mnist"

# Create the dataset directory if it doesn't exist
mkdir -p $DATASET_DIR

# Array of MNIST dataset files
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST dataset..."

# Retry function for download
download_with_retry() {
    local file=$1
    local url=$2
    local output=$3
    local retries=3
    local count=0

    while [ $count -lt $retries ]; do
        wget -q --show-progress "$url" -O "$output" && return 0
        count=$((count + 1))
        echo "Retry $count/$retries for $file..."
    done

    echo "Failed to download $file after $retries attempts."
    return 1
}

# Download and extract each file
for FILE in "${FILES[@]}"; do
    FILE_PATH="$DATASET_DIR/$FILE"
    EXTRACTED_FILE="${FILE%.gz}"

    # Check if the file is already downloaded and valid
    if [ -f "$FILE_PATH" ]; then
        echo "$FILE already exists, verifying integrity..."
        if ! gzip -t "$FILE_PATH" &>/dev/null; then
            echo "File $FILE is corrupted. Re-downloading..."
            rm -f "$FILE_PATH"
        else
            echo "$FILE is valid. Skipping download."
            continue
        fi
    fi

    # Download the file with retries
    echo "Downloading $FILE..."
    if ! download_with_retry "$FILE" "$MNIST_URL_BASE/$FILE" "$FILE_PATH"; then
        echo "Aborting script. Please check your network connection."
        exit 1
    fi

    # Extract the file
    echo "Extracting $FILE..."
    gunzip -kf "$FILE_PATH"

    # Verify extraction success
    if [ ! -f "$DATASET_DIR/$EXTRACTED_FILE" ]; then
        echo "Failed to extract $FILE. Please check gzip."
        exit 1
    fi
done

echo "MNIST dataset downloaded and extracted to $DATASET_DIR."
