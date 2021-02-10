# %%
from datetime import datetime

# %%
def log_error(path, message): 
    now = datetime.now()
    with open('error.log', 'a') as file:
        file.write(now.strftime("%Y/%m/%d, %H:%M:%S") + message + '\n') 