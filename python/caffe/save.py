def save(fileName,var1, var2, version):
    file = open(fileName)#, mode, buffering, encoding, errors, newline, closefd, opener)
    file.write(var1)
    file.write(var2)
    file.write(version)
    file.close()
    return