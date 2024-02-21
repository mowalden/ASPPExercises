"""Creates various classes corresponding to members of a classroom.

Typical usage example:

  moritz = Person('Moritz', 'Walden')
  moritz.get_name()
"""

class Person():
    """Person class describing a person with a name

    Attributes:
        first_name: The first name of the person.
        last_name: The last name of the person.
    """

    def __init__(self, first_name, last_name):
        """Initializes the instance based on name.

        Args:
            first_name: The first name of the person.
            last_name: The last name of the person.
        """
        self.first_name = first_name
        self.last_name = last_name

    def get_name(self):
        """Prints the full name.

        Args:
            first_name: The first name of the person.
            last_name: The last name of the person.

        Returns:
            Full name as a string.
        """
        return f'{self.first_name} {self.last_name}'

class Student(Person):
    """Student class describing a student with a name and subject area.

    Attributes:
        first_name: The first name of the student.
        last_name: The last name of the student.
        subject_area: Subject are of the student.
    """
    def __init__(self, first_name, last_name, subject_area):
        """Initializes the instance based on name and subject area.

        Args:
            first_name: The first name of the person.
            last_name: The last name of the person.
            subject_area: Subject are of the student.
        """
        super().__init__(first_name, last_name)
        self.subject_area = subject_area

    def print_name_and_subject(self):
        """Prints the full name and subject area.

        Args:
            first_name: The first name of the person.
            last_name: The last name of the person.
        """
        print(f'Name: {self.get_name()} \t Subject area: {self.subject_area}')


class Teacher(Person):
    """Teacher class describing a teacher with a name and class.

    Attributes:
        first_name: The first name of the teacher.
        last_name: The last name of the teacher.
        class: Class that they are teaching.
    """
    def __init__(self, first_name, last_name, class_name):
        """Initializes the instance based on name and class.

        Args:
            first_name: The first name of the teacher.
            last_name: The last name of the teacher.
            class_name: Class that they are teaching.
        """
        super().__init__(first_name, last_name)
        self.class_name = class_name

    def print_name_and_class(self):
        """Prints the full name and class.

        Args:
            first_name: The first name of the teacher.
            last_name: The last name of the teacher.
            class_name: Class that they are teaching.
        """
        print(f'Name: {self.get_name()} \t Class: {self.class_name}')