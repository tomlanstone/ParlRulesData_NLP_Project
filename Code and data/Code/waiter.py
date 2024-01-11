import datetime
import time

def wait_until(target_hour, target_minute):
    while True:
        now = datetime.datetime.now()
        if now.hour == target_hour and now.minute == target_minute:
            break
        time.sleep(60)

def get_time():
    now = datetime.datetime.now()
    return now.strftime('%H:%M:%S')