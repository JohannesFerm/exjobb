import rclpy
from rclpy.node import Node
from flask import Flask, request
import threading

from std_msgs.msg import String
from hqv_public_interface.msg import RemoteDriverDriveCommand


class PoseDrive(Node):

    def __init__(self):
        super().__init__('motorController')

        self.drive_publisher = self.create_publisher(RemoteDriverDriveCommand, '/hqv_mower/remote_driver/drive', 100)

        self.app = Flask(__name__)
        self.command = ""

        @self.app.route('/command', methods=['POST'])
        def receive_message():
            # Get the message from the POST request
            self.command = request.json.get('command')  # Match the key used in the client
            print(f"Received command: {self.command}")
            self.motorControl(self.command)
            
            return {"status": "success", "message": "Message received"}, 200
        
        flask_thread = threading.Thread(target=self.run_flask)
        flask_thread.daemon = True
        flask_thread.start()

    def run_flask(self):
        self.app.run(debug=False, host='0.0.0.0', port=5000)

    def motorControl(self, msg): #Inspiration from the provided remote_drive_node.py
        if self.command == "left":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 0.33
            msg.steering = 2.0
            self.drive_publisher.publish(msg)
        elif self.command == "right":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 0.33
            msg.steering = -2.0
            self.drive_publisher.publish(msg)
        elif self.command == "closedhand":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 0.33
            msg.steering = 0.0
            self.drive_publisher.publish(msg)
        elif self.command == "openhand":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = -0.33
            msg.steering = 0.0
            self.drive_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    motorController = PoseDrive()

    rclpy.spin(motorController)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    motorController.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()