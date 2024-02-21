class Mammals:
    ''' Constructor for this class. '''
    def __init__(self):
        # Create some member animals
        self.members = ['Tiger', 'Lion', 'Icebear']
    
    def printMembers(self):
        print('Printing dangerous members of the Mammals class')
        for member in self.members:
            print('\t%s' % member)