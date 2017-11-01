

def print_parameters(**kwargs):
    print("\n######################################################")
    print("Parameters:")
    for k,v in kwargs.iteritems():
         print "%s = %s" % (k, v)
    print("######################################################\n")
