import os
import shutil
import argparse
from git import Repo

def cloning_data_provider(git_url):
    return git_url, git_url.split("/")[-1].split(".git")[0]

def clone_repo(git_url, clone_dir):
    # Clone the repository to the specified directory
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    Repo.clone_from(git_url, clone_dir)
    print(f"Cloned {git_url} to {clone_dir}")
    repo_name = git_url.split("/")[-1].split(".git")[0]
    return repo_name

def flatten_repo(repo_dir, target_dir):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Walk through the repository directory
    for root, _, files in os.walk(repo_dir):
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            # Skip .git directory
            if '.git' in file_path:
                continue
            # Calculate the destination path with .txt extension
            dest_path = os.path.join(target_dir, file + ".txt")

            # Handle potential file name conflicts by renaming
            base, extension = os.path.splitext(file)
            counter = 1
            while os.path.exists(dest_path):
                new_file_name = f"{base}_{counter}.txt"
                dest_path = os.path.join(target_dir, new_file_name)
                counter += 1

            # Move file to target directory and rename with .txt extension
            shutil.move(file_path, dest_path)
            print(f"Moved {file_path} to {dest_path}")

    # Remove the original repository directory
    shutil.rmtree(repo_dir)
    print(f"Removed original repository directory {repo_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone a Git repository, flatten it, and move all files to a single directory.")
    parser.add_argument("git_url", type=str, help="URL of the Git repository to clone")
    args = parser.parse_args()

    git_url = args.git_url

    # Define paths
    repo_dir = clone_repo(*cloning_data_provider(git_url))  # Replace with the name of your cloned repository directory
    target_dir =  "_"+repo_dir

    # Flatten the repository
    flatten_repo(repo_dir, target_dir)

    print("All files have been moved to a single directory and original repository directory has been removed.")
