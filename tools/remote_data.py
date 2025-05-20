import os
import paramiko
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


# def download_from_remote(remote_dir, local_dir):
#     # 连接 SSH
#     transport = paramiko.Transport((hostname, port))
#     transport.connect(username=username, password=password)
#     sftp = paramiko.SFTPClient.from_transport(transport)

#     file_list = sorted(sftp.listdir(remote_dir))

#     for file in tqdm(file_list, desc="Downloading files"):
#         remote_path = f"{remote_dir.rstrip('/')}/{file}"
#         local_path = os.path.join(local_dir, file)
#         try:
#             if not os.path.exists(local_path):
#                 sftp.get(remote_path, local_path)
#             else:
#                 print(f"Skipped {file}, already exists.")
#         except FileNotFoundError as e:
#             print(f"[Warning] File not found on remote: {remote_path}")
#         except Exception as e:
#             print(f"[Error] Failed to download {remote_path}: {e}")


# def upload_to_remote(remote_dir, local_dir, hostname, port, username, password):
#     # 建立SSH连接
#     transport = paramiko.Transport((hostname, port))
#     transport.connect(username=username, password=password)
#     sftp = paramiko.SFTPClient.from_transport(transport)

#     # 获取本地文件列表
#     file_list = sorted(glob.glob(local_dir + '/*'))

#     for file in tqdm(file_list, desc="Uploading files"):
#         filename = os.path.basename(file)
#         remote_path = f"{remote_dir.rstrip('/')}/{filename}"
#         try:
#             # 检查远程文件是否已存在
#             try:
#                 sftp.stat(remote_path)
#                 print(f"Skipped {filename}, already exists.")
#                 continue
#             except FileNotFoundError:
#                 pass

#             # 上传
#             sftp.put(file, remote_path)
#         except Exception as e:
#             print(f"[Error] Failed to upload {filename}: {e}")

#     sftp.close()
#     transport.close()


def download_file(sftp, remote_path, local_path):
    try:
        if os.path.exists(local_path):
            return f"Skipped {os.path.basename(remote_path)}, already exists."
        sftp.get(remote_path, local_path)
        return f"Downloaded {os.path.basename(remote_path)}"
    except Exception as e:
        return f"[Error] {remote_path}: {e}"

def download_from_remote(remote_dir, local_dir, hostname, port, username, password, max_workers=8):
    # 建立 SSH 和 SFTP 连接
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # 获取远程文件列表
    remote_file_list = sftp.listdir(remote_dir)
    remote_paths = [f"{remote_dir.rstrip('/')}/{file}" for file in remote_file_list]

    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for remote_file in remote_paths:
            local_file = f"{local_dir}/{os.path.basename(remote_file)}"
            futures.append(executor.submit(download_file, sftp, remote_file, local_file))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            print(future.result())

    sftp.close()
    transport.close()


def upload_file(sftp, local_file, remote_path):
    filename = os.path.basename(local_file)
    try:
        try:
            sftp.stat(remote_path)
            return f"Skipped {filename}, already exists."
        except FileNotFoundError:
            sftp.put(local_file, remote_path)
            return f"Uploaded {filename}"
    except Exception as e:
        return f"[Error] {filename}: {e}"

def upload_to_remote(remote_dir, local_dir, hostname, port, username, password, max_workers=8):
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    file_list = sorted(glob.glob(os.path.join(local_dir, '*')))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in file_list:
            filename = os.path.basename(file)
            remote_path = f"{remote_dir.rstrip('/')}/{filename}"
            futures.append(executor.submit(upload_file, sftp, file, remote_path))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            print(future.result())

    sftp.close()
    transport.close()


if __name__ == '__main__':
    # 配置信息
    hostname = '111.111.111.111'
    port = 22
    username = 'yzhen'
    password = 'yzhen'
    
    # 上传文件
    remote_dir = '/data0/yzhen/data/totalseg_v201'
    local_dir  = 'D:/codes/data/Totalsegmentator_dataset_v201'
    upload_to_remote(remote_dir, local_dir, hostname, port, username, password)