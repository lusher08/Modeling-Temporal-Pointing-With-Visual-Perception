import pygame
import time
import csv

n_flashes = 5
flash_interval = 3.0
flash_duration = 0.5

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Flash Click Simulation")

flash_times = []
click_times = []

clock = pygame.time.Clock()
t0 = time.time() #기준 시간
flash_start = t0

#======================플래시 시뮬레이션===========================
for i in range(n_flashes):
    #다음 플래시까지 대기
    while time.time() - flash_start < flash_interval:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    click_time = time.time() - t0  
                    click_times.append(click_time)
        clock.tick(60)

    #플래시 시작
    screen.fill((255, 255, 255))
    pygame.display.flip()
    flash_start = time.time()
    flash_times.append(flash_start - t0)   

    while time.time() - flash_start < flash_duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    click_time = time.time() - t0
                    click_times.append(click_time)
        clock.tick(60)

    #플래시 끝
    screen.fill((0, 0, 0))
    pygame.display.flip()

pygame.quit()

#======================출력 및 저장===========================
print("\nClick - Flash (s):")
for i, flash_time in enumerate(flash_times):
    if click_times:
        closest_click = min(click_times, key=lambda x: abs(x - flash_time))
        delta = closest_click - flash_time
        print(f"Flash {i+1}: {flash_time:.2f}s, Closest Click: {closest_click:.2f}s, Δ={delta:.2f}s")
    else:
        print(f"Flash {i+1}: {flash_time:.2f}s, No clicks")

#CSV 저장
with open("human_data.csv", "a", newline="") as f:
    writer = csv.writer(f)
    for flash_time in flash_times:
        if click_times:
            closest_click = min(click_times, key=lambda x: abs(x - flash_time))
        else:
            closest_click = ""
        writer.writerow([flash_time, closest_click])
print("Human data saved to human_data.csv")
