import os 

def find_file(file_extension, search_path):
  if not os.path.exists(search_path):
    print(f"Folder ({search_path}) not found")
    return None
  
  files_in_path = os.listdir(search_path)
  
  for file in files_in_path:
    print(file)
    if file.endswith(file_extension):
      return os.path.join(search_path, file)
  
  return None