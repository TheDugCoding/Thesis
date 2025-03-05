print("finally it works")

# Define the filename
filename = "here I am!.txt"

# Open the file in write mode, creating it if it doesn't exist
with open(filename, 'w') as file:
    # Write some text to the file
    file.write("Hello, this is a simple text file!")

print(f"File '{filename}' has been created and written to.")