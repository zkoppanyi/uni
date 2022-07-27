import subprocess

def get_exif_data(img_path):

    exif_out = subprocess.run(['exiftool', img_path], capture_output=True, text=True).stdout
    exif_data = {}
    for line in exif_out.split('\n'):
        elems = line.split(':')
        if len(elems) > 1:
            key = elems[0].strip()
            value = ':'.join(elems[1:]).strip()
            try:
                value = float(value)
            except ValueError:
                value = value
            exif_data[key] = value
    return exif_data
