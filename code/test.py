import multiprocessing

# Define a function for the first process
def process_function_1():
    for i in range(5):
        print("Process 1: Iteration", i)

# Define a function for the second process
def process_function_2():
    for i in range(5):
        print("Process 2: Iteration", i)

if __name__ == "__main__":
    # Create two process objects
    process1 = multiprocessing.Process(target=process_function_1)
    process2 = multiprocessing.Process(target=process_function_2)

    # Start the processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()

    # Code here will only run after both processes have finished
    print("Both processes have finished")
