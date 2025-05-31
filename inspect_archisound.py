from archisound import ArchiSound
import inspect

# Create an instance
model = ArchiSound()

# Print the class attributes and methods
print("ArchiSound class attributes and methods:")
for name, value in inspect.getmembers(ArchiSound):
    if not name.startswith('_'):  # Skip private attributes
        print(f"{name}: {type(value)}")

# Print instance attributes and methods
print("\nArchiSound instance attributes and methods:")
for name, value in inspect.getmembers(model):
    if not name.startswith('_'):  # Skip private attributes
        print(f"{name}: {type(value)}")

# Print the docstring if available
if ArchiSound.__doc__:
    print("\nDocstring:")
    print(ArchiSound.__doc__)
