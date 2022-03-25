def exhaustive_list(format_list):
    return format_list

def range_list(start, stop, step=1):
    return list(range(start, stop, step))


SUPPORTED_FORMAT_LIST = {'range': range_list,
                         'exhaustive': exhaustive_list}

def get_format_list(format_list_dict):
    format_list_type = format_list_dict.pop('type')
    return SUPPORTED_FORMAT_LIST[format_list_type](**format_list_dict)


if __name__ == '__main__':
    print(range_list(5, 0, -1))

