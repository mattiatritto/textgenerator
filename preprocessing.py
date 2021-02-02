import os



"""Delete lines from a file that contains a given word / sub-string """
def delete_line_with_word(file_name, word):
    delete_line_by_condition(file_name, lambda x : word in x )



""" In a file, delete the lines at line number in given list"""
def delete_line_by_condition(original_file, condition):

    dummy_file = original_file + '.bak'
    is_skipped = False

    # Open original file in read only mode and dummy file in write mode
    with open(original_file, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Line by line copy data from original file to dummy file
        for line in read_obj:
            # if current line matches the given condition then skip that line
            if condition(line) == False:
                write_obj.write(line)
            else:
                is_skipped = True
                
    # If any line is skipped then rename dummy file as original file
    if is_skipped:
        os.remove(original_file)
        os.rename(dummy_file, original_file)
    else:
        os.remove(dummy_file)