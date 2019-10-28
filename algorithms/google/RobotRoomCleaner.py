

class Robot:
    def clean(self):
        pass

    def move(self):
        pass

    def turnLeft(self):
        pass

    def turnRight(self):
        pass


class RobotRoomCleaner(object):
    def cleanRoom(self, robot):
        # 4 directions: top, right, bottom, left
        dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))
        visited = set()

        def helper(x, y):
            robot.clean()
            visited.add((x, y))
            for dir in dirs:
                nx, ny = x + dir[0], y + dir[1]
                # Robot can move to next position
                if (nx, ny) not in visited and robot.move():
                    helper(nx, ny)
                    robot.turnRight()
                    robot.turnRight()
                    robot.move()
                    robot.turnLeft()
                    robot.turnLeft()
                # Turn to next direction
                robot.turnRight()

        helper(0, 0)
