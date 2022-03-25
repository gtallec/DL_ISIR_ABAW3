import json

def dict_from_json(json_file):
    with open(json_file) as f:
        json_dict = json.load(f)
    return json_dict


if __name__ == '__main__':
    pass






    


    



