import socket
import time

def send_udp_signal(ip, port, message):
    """
    지정된 IP 주소와 포트로 UDP 신호를 보내는 함수.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (ip, port))
    sock.close()

def analyze_eeg_data():
    # EEG 신호를 분석하여 0, 1, 2 중 하나의 결과를 3초마다 반환하는 함수
    result = 0
    while True:
        yield result
        result = (result + 1) % 3
        time.sleep(3)  # 3초 대기

def main():
    udp_ip = "127.0.0.1"  # 유니티가 실행되고 있는 IP 주소
    udp_port = 9876  # 유니티에서 설정한 포트 번호

    eeg_data_generator = analyze_eeg_data()

    while True:
        result = next(eeg_data_generator)
        send_udp_signal(udp_ip, udp_port, str(result))
        print("Sent UDP signal to {}:{} with result '{}'".format(udp_ip, udp_port, result))
        time.sleep(1)  # 1초마다 데이터를 전송합니다.

if __name__ == '__main__':
    main()
