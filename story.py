from image_match.goldberg import ImageSignature

gis = ImageSignature()

from Violet.Violet_base import list_all_files,get_parent_and_grandparent_dir
import shutil
import os

def copy_file_shutil(source_path, destination_path):
  """
  使用 shutil.copy2() 函数复制文件。

  Args:
      source_path (str): 源文件路径。
      destination_path (str): 目标文件路径。
  """
  try:
    shutil.copy2(source_path, destination_path)
    print(f"File copied from {source_path} to {destination_path}")
    return True
  except FileNotFoundError:
     print(f"Error: Source file not found at {source_path}")
  except Exception as e:
     print(f"Error: Failed to copy file. {e}")

def get_story(button_path, image_path, output_path):
    button_list = list_all_files(button_path)

    std = gis.generate_signature('000008.jpg')

    for path in button_list:
        button = gis.generate_signature(path)
        distance = gis.normalized_distance(button, std)
        if distance <= 0.4:
            base_name, parent_name, grandparent_name = get_parent_and_grandparent_dir(path)
            if os.path.isdir(output_path + '/' + parent_name) == False:
                os.makedirs(output_path + '/' + parent_name)
            r = copy_file_shutil(image_path + '/' + parent_name + '/' + base_name, output_path + '/' + parent_name + '/' + base_name)
            if r == True:
                print(f"{base_name} copied to {output_path + '/' + parent_name}")

if __name__ == '__main__':
    button_path = 'E:/Genshin_impact_circle/'
    image_path = 'E:/Genshin_Impact/'
    output_path = 'E:/Genshin_Impact_story/'
    get_story(button_path, image_path, output_path)
