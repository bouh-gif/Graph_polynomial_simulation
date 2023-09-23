# import sys
log_file = open('myprog.log', 'w')
def log(text):
    print(text)
    log_file.write(str(text) + '\n')


log('hello')
log("test")
