import os
import tiktoken
import glob
import argparse

# Directories to skip
EXCLUDED_DIRS = [
    '.github', '.idea', '__pycache__', '.pytest_cache', '.git',
    'venv', 'env', '.venv', '.env', 'build', 'dist', 'node_modules'
]

def should_skip_path(path):
    """Check if the path should be skipped based on excluded directories."""
    parts = path.split(os.sep)
    return any(excluded in parts for excluded in EXCLUDED_DIRS)

def count_tokens_in_file(file_path, encoding_name="cl100k_base"):
    """Count tokens in a single file using the specified encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(content)
        return len(tokens)
    except UnicodeDecodeError:
        # Skip binary files or files with unknown encoding
        print(f"Skipping (not text file): {file_path}")
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def estimate_tokens_in_directory(directory_path, encoding_name="cl100k_base"):
    """Estimate tokens for all files in the directory, skipping excluded directories."""
    total_tokens = 0
    file_counts = {}
    file_types = {}
    
    skip_extensions = ['.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.inv', '.npz', 
                       '.xyz', '.h5md', '.h5',
                        '.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz', '.bz2', '.xz', 
                        '.pickle', '.pkl', '.jsonl', '.parquet', '.msgpack', '.xlsx', '.xls',
                        '.docx', '.pptx', '.pdf', '.epub', '.mobi', '.azw3', '.epub3',
                        '.doc', '.ppt', '.xls', '.csv', '.tsv', '.xml', '.html', '.htm',
                        '.css', '.js', '.json', '.yaml', '.yml', '.toml', '.ini',]
    
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip hidden files and certain extensions we know are binary
            _, ext = os.path.splitext(file)
            if file.startswith('.') or ext.lower() in skip_extensions:
                continue
                
            token_count = count_tokens_in_file(file_path, encoding_name)
            
            if token_count > 0:
                total_tokens += token_count
                file_counts[file_path] = token_count
                
                # Track statistics by file extension
                ext = ext.lower() if ext else '(no extension)'
                if ext not in file_types:
                    file_types[ext] = {'count': 0, 'tokens': 0}
                file_types[ext]['count'] += 1
                file_types[ext]['tokens'] += token_count
        
    return total_tokens, file_counts, file_types

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Estimate token count in all text files.')
    parser.add_argument('project_dir', type=str, help='Path to the project directory')
    
    args = parser.parse_args()
    
    # Validate the directory exists
    if not os.path.isdir(args.project_dir):
        print(f"Error: Directory '{args.project_dir}' does not exist.")
        exit(1)
    
    print(f"Analyzing files in: {args.project_dir}")
    print(f"Excluding directories: {', '.join(EXCLUDED_DIRS)}")
    
    total, file_details, file_types = estimate_tokens_in_directory(args.project_dir)
    
    if len(file_details) == 0:
        print(f"No readable text files found in {args.project_dir}")
        exit(0)
    
    print(f"\nTotal estimated tokens: {total:,}")
    print(f"Number of files processed: {len(file_details)}")
    print(f"Average tokens per file: {total / len(file_details):,.1f}")
    
    # Print file type statistics
    print("\nToken count by file type:")
    sorted_types = sorted(file_types.items(), key=lambda x: x[1]['tokens'], reverse=True)
    for ext, stats in sorted_types:
        print(f"{ext}: {stats['tokens']:,} tokens in {stats['count']} files")
    
    # Print details of the largest files
    sorted_files = sorted(file_details.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 largest files by token count:")
    for path, count in sorted_files[:10]:
        print(f"{path}: {count:,} tokens")