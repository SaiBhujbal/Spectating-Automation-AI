import socket
import pyautogui

def server_program():
    host = ''  # Listen on all network interfaces
    port = 5000  # Port to listen on

    server_socket = socket.socket()  # Create a socket
    server_socket.bind((host, port))  # Bind the socket to the host and port
    server_socket.listen(2)  # Start listening for connections
    print("Server is listening on port", port)

    while True:
        conn, address = server_socket.accept()  # Accept a new connection
        print("Connection from:", address)

        # Keep the connection open to handle multiple messages
        while True:
            try:
                data = conn.recv(1024).decode()  # Receive data from the client
                if not data:
                    # If no data is received, break from the loop to close the connection
                    break
                print("Received from client:", data)

                # Simulate key press using pyautogui
                pyautogui.press(data)

                # Optional: Send a confirmation back to the client
                conn.sendall("Key pressed: {}".format(data).encode())
            except Exception as e:
                print("Error:", e)
                break  # Break from the loop to close the connection in case of an error

        conn.close()  # Close the connection after handling all messages
        print("Connection closed.")

if __name__ == '__main__':
    server_program()    