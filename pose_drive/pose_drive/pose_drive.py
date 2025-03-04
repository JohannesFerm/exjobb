import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from hqv_public_interface.msg import RemoteDriverDriveCommand


class PoseDrive(Node):

    def __init__(self):
        super().__init__('motorController')
        self.subscriber = self.create_subscription(String, 'humanpose', self.motorControl, 10)
        self.drive_publisher = self.create_publisher(RemoteDriverDriveCommand, '/hqv_mower/remote_driver/drive', 100)


    def motorControl(self, msg): #Inspiration from the provided remote_drive_node.py
        if msg.data == "left":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 1.0
            msg.steering = 2.0
            self.drive_publisher.publish(msg)
        elif msg.data == "right":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 1.0
            msg.steering = -2.0
            self.drive_publisher.publish(msg)
        elif msg.data == "forward":
            msg = RemoteDriverDriveCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.speed = 0.33
            msg.steering = 0.0
            self.drive_publisher.publish(msg)
        elif msg.data == "backward":
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