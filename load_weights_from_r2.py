import boto3
import os


def load_weights_from_r2(
    bucket_name: str = "ratemepls-weights",
    local_dir: str = "./weights",
) -> str:
    """
    Download model weights from Cloudflare R2 bucket.
    
    Args:
        bucket_name: Name of the R2 bucket
        local_dir: Local directory to save the weights
    
    Environment Variables Required:
        R2_ACCOUNT_ID: Cloudflare account ID
        R2_ACCESS_KEY_ID: R2 access key ID
        R2_SECRET_ACCESS_KEY: R2 secret access key
        R2_WEIGHTS_PREFIX: Prefix/folder path in the bucket (e.g., "p2zil2wb_weights/")
    
    Returns:
        Path to the local directory containing the downloaded weights
    
    Example:
        >>> weights_dir = load_weights_from_r2()
        >>> model.load_finetune_weights(weights_dir)
    """
    # Get credentials and config from environment variables
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    weights_prefix = os.environ.get("R2_WEIGHTS_PREFIX")
    
    if not all([account_id, access_key_id, secret_access_key, weights_prefix]):
        raise ValueError(
            "R2 credentials not provided. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY, and R2_WEIGHTS_PREFIX environment variables."
        )
    
    # Create R2 endpoint URL
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    # Initialize S3 client (R2 is S3-compatible)
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",  # R2 uses "auto" as region
    )
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # List all objects in the bucket with the given prefix
    print(f"Listing objects in bucket '{bucket_name}' with prefix '{weights_prefix}'...")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=weights_prefix)
    
    if "Contents" not in response:
        raise FileNotFoundError(
            f"No objects found in bucket '{bucket_name}' with prefix '{weights_prefix}'"
        )
    
    # Download each file
    downloaded_files = []
    for obj in response["Contents"]:
        key = obj["Key"]
        
        # Skip if it's just a directory marker
        if key.endswith("/"):
            continue
        
        # Get the relative path (remove the prefix)
        relative_path = key[len(weights_prefix):] if key.startswith(weights_prefix) else key
        
        # Skip empty relative paths
        if not relative_path:
            continue
        
        # Create local file path
        local_file_path = os.path.join(local_dir, relative_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        print(f"Downloading {key} -> {local_file_path}")
        s3_client.download_file(bucket_name, key, local_file_path)
        downloaded_files.append(local_file_path)
    
    print(f"Successfully downloaded {len(downloaded_files)} files to {local_dir}")
    return local_dir
