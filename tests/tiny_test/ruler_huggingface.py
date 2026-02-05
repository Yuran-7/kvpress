from huggingface_hub import list_repo_files

files = list_repo_files("simonjegou/ruler", repo_type="dataset")
# 看看有没有以 "4096/" 开头的文件
print([f for f in files if f.startswith("4096/")][:50])
