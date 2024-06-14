import socket

def start_udp_server(ip, port):
    """
    지정된 IP 주소와 포트에서 UDP 신호를 수신하는 함수.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    print("Listening on {}:{}".format(ip, port))

    while True:
        data, addr = sock.recvfrom(1024)  # 버퍼 크기는 1024 바이트
        print("Received message: {} from {}".format(data.decode(), addr))

if __name__ == '__main__':
    udp_ip = '0.0.0.0'  # 모든 네트워크 인터페이스에서 수신
    udp_port = 9876  # 유니티에서 설정한 포트 번호

    start_udp_server(udp_ip, udp_port)
