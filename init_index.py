from config import index_client, settings

if __name__ == "__main__":
    index_client(create=True)
    cfg = settings()
    print(f"Ready: {cfg['index_name']} / {cfg['namespace']} (1536 dimensions, Serverless)")
