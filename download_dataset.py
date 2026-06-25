import kagglehub

print("Downloading bigearthnet-14k dataset using kagglehub...")
path = kagglehub.dataset_download("narendraaironi/bigearthnet-14k")
print("Path to dataset files:", path)
