import sys

def get_error_details(error):
    _, _, tb = sys.exc_info()
    if tb is not None:
        file_name = tb.tb_frame.f_code.co_filename
        line_no = tb.tb_lineno
        return f"Error occurred in {file_name} at line {line_no} : {str(error)}"
    return str(error)

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message)

    def __str__(self):
        return self.error_message

