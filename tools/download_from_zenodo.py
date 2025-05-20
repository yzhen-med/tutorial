import os
import requests
import subprocess

ACCESS_TOKEN = "UDkdBAA1qbGEnyqf4oJALPnslJ7S4sEYmOVRiv093Pe52LrF5q8THS5WzGyi"
record_id = "14223624"  # LUNA25 record id

output_folder = "/data0/yzhen/data/LN25"
os.makedirs(output_folder, exist_ok=True)

# 获取 Zenodo 记录的元数据
r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
if r.status_code != 200:
    print("Error retrieving record:", r.status_code, r.text)
    exit()

# 提取下载 URL 和文件名
download_urls = [f['links']['self'] for f in r.json()['files']]
filenames = [f['key'] for f in r.json()['files']]

print(f"Total files to download: {len(download_urls)}")

# 使用 Aria2 多线程下载
for filename, url in zip(filenames, download_urls):
    file_path = os.path.join(output_folder, filename)
    print(f"Downloading: {filename} -> {file_path}")

    cmd = f"aria2c -x 16 -s 16 -j 1 --out {file_path} --dir {output_folder} \"{url}?access_token={ACCESS_TOKEN}\""
    subprocess.run(cmd, shell=True)

print("All downloads completed successfully!")