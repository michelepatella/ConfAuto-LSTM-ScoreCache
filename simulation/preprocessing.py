import math

def preprocess_data(row):
    # get the requested key
    key = row['request']

    # calculate timestamps in seconds
    angle = math.atan2(row['sin_time'], row['cos_time'])
    if angle < 0:
        angle += 2 * math.pi
    current_time = angle / (2 * math.pi) * 24 * 3600

    return key, current_time