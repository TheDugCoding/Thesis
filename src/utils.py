import os

'''
File used for relative paths useful when the code is run in different machines
'''



def find_ancestor_folder(target_folder):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Start from script's directory

    while True:
        if os.path.basename(current_dir) == target_folder:  # Check if folder name matches
            return current_dir
        parent_dir = os.path.dirname(current_dir)  # Move up one level
        if parent_dir == current_dir:  # If reached root, stop searching
            raise FileNotFoundError(f"Folder '{target_folder}' not found in ancestors.")
        current_dir = parent_dir  # Continue searching upwards

def get_data_folder():
    return data_folder

def get_src_folder():
    return src_folder

def get_data_sub_folder(folder_name):
    return os.path.join(data_folder, folder_name)

def get_src_sub_folder(folder_name):
    return os.path.join(src_folder, folder_name)

main_folder = find_ancestor_folder("Thesis")
data_folder = os.path.join(main_folder, 'data')
src_folder = os.path.join(main_folder, 'src')