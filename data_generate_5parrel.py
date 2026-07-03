# generate 5st filter
# need project10.aedt
# change the parameters in the project10.aedt
# 3parrel



import os
import random
import time
import traceback
import getpass
import shutil
import numpy as np
import multiprocessing
from ansys.aedt.core import Desktop, Hfss

# ==========================================
# 0. 仿真配置
# ==========================================
NUM_WORKERS = 3  # 并行运行的进程数量
CORES_PER_TASK = 16  # 单实例运行分配的 CPU 核心数 (3个进程将占用 3 * 16 = 48 核心)
SKIP_VALIDATION = True

# 环境配置
os.environ["ANSYSEM_ROOT241"] = "/tools/ansys/ansysem/2024R1/v241/Linux64/"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["ANSYSEM_ROOT241"]

# 目标工程路径
project_dir = "/home/users/YaoYuming/.mw/Desktop/"
project_name = "Project10"
project_path = os.path.join(project_dir, f"{project_name}.aedt")
export_folder = os.path.join(project_dir, "dataset_surrogate_model_523")

if not os.path.exists(export_folder):
    os.makedirs(export_folder)


# ==========================================
# 辅助函数：深度清理与环境准备
# ==========================================
def clean_environment(proj_path):
    """仅在程序一开始调用：清理属于当前用户的僵尸进程以及残留的锁文件"""
    current_user = getpass.getuser()
    print(f"[*] 正在扫描并清理属于 {current_user} 的后台僵尸 Ansys 进程...")

    # 强制结束当前用户的 ansysedt 和相关无头进程
    os.system(f"pkill -9 -u {current_user} -f ansysedt > /dev/null 2>&1")
    os.system(f"pkill -9 -u {current_user} -f ansoftdxe > /dev/null 2>&1")
    time.sleep(3)  # 等待操作系统释放端口

    # 清理母工程的锁文件
    lock_file = proj_path + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception:
            pass


def duplicate_project(original_path, worker_id):
    """为每个 Worker 复制一份独立的工程文件，避免读写冲突"""
    base_dir = os.path.dirname(original_path)
    base_name = os.path.splitext(os.path.basename(original_path))[0]

    new_name = f"{base_name}_worker{worker_id}"
    new_proj_path = os.path.join(base_dir, f"{new_name}.aedt")

    # 复制 .aedt 文件
    print(f"[*] 正在为 Worker {worker_id} 生成独立工程: {new_proj_path}")
    shutil.copy2(original_path, new_proj_path)

    # 如果存在对应的 .aedb 文件夹，也一并复制
    original_aedb = os.path.join(base_dir, f"{base_name}.aedb")
    new_aedb = os.path.join(base_dir, f"{new_name}.aedb")
    if os.path.exists(original_aedb):
        if os.path.exists(new_aedb):
            shutil.rmtree(new_aedb)
        shutil.copytree(original_aedb, new_aedb)

    return new_proj_path


# ==========================================
# 核心工作函数 (将被多进程调用)
# ==========================================
def run_simulations_worker(worker_id, tasks, proj_path, export_dir):
    # 为当前 Worker 创建独立的日志文件，避免多进程写入冲突
    log_file_path = os.path.join(export_dir, f"final_data_log_worker{worker_id}.csv")

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("filename,split,l1,l2,l3,g1,g2\n")

    total_samples = len(tasks)
    print(f"[Worker {worker_id}] 启动任务队列。当前进程负责数量: {total_samples}")

    RESTART_THRESHOLD = 150
    app = None
    desktop = None

    def start_hfss_session():
        nonlocal app, desktop
        print(f"\n[Worker {worker_id}] 正在初始化 Desktop 并启动 HFSS 会话...")

        # new_desktop_session=True 会自动寻找空闲端口，适合多进程
        desktop = Desktop(
            version="2024.1",
            non_graphical=True,
            new_desktop_session=True
        )
        time.sleep(5)

        app = Hfss(project=proj_path, version="2024.1")

        if getattr(app, "oproject", None) is None:
            raise RuntimeError(f"[Worker {worker_id}] HFSS 工程加载失败！")

        if not app.design_list:
            raise RuntimeError(f"[Worker {worker_id}] 在工程中没有找到任何 Design！")

        app.set_active_design(app.design_list[0])

        setup_name = app.setup_names[0] if app.setup_names else None
        sweep_name = app.get_sweeps(setup_name)[0] if setup_name and app.get_sweeps(setup_name) else None

        if not setup_name or not sweep_name:
            raise RuntimeError(f"[Worker {worker_id}] 没有找到 Setup 或 Sweep。")

        print(f"[Worker {worker_id}] HFSS 会话启动成功！")
        return setup_name, sweep_name

    def close_hfss_session():
        nonlocal app, desktop
        if app is not None or desktop is not None:
            print(f"[Worker {worker_id}] 释放内存，关闭 HFSS 会话...")
            try:
                if app is not None:
                    app.release_desktop(close_projects=True, close_desktop=True)
            except Exception as e:
                print(f"[Worker {worker_id}] 关闭会话时警告: {str(e)}")
            finally:
                app = None
                desktop = None
            time.sleep(5)

    try:
        active_setup, active_sweep = start_hfss_session()
        solution_full_name = f"{active_setup} : {active_sweep}"

        for i, task in enumerate(tasks):
            if i > 0 and i % RESTART_THRESHOLD == 0:
                print(f"\n[Worker {worker_id}] 达到 {RESTART_THRESHOLD} 次计算，重启 Desktop 清理内存...")
                close_hfss_session()
                # 注意：这里千万不能调 clean_environment(pkill)，否则会杀掉其他 Worker！
                active_setup, active_sweep = start_hfss_session()

            start_time = time.time()
            split = task["split"]
            global_id = task["global_id"]

            print(f"[Worker {worker_id}] Task {i + 1}/{total_samples} (Global #{global_id})")

            new_vars = {
                "l1": f"{task['l1']}mm",
                "l2": f"{task['l2']}mm",
                "l3": f"{task['l3']}mm",
                "g1": f"{task['g1']}mm",
                "g2": f"{task['g2']}mm"
            }
            for k, v in new_vars.items():
                app[k] = v

            if not SKIP_VALIDATION:
                if not app.validate_full_design()[1]:
                    print(f"[Worker {worker_id}] Validation FAILED for Global #{global_id}. Skipping...")
                    continue

            success = app.analyze_setup(active_setup, cores=CORES_PER_TASK)
            if not success:
                print(f"[Worker {worker_id}] Error: Simulation failed for Global #{global_id}")
                continue

            sol_data = app.post.get_solution_data(
                expressions=["re(S(1,1))", "im(S(1,1))", "re(S(2,1))", "im(S(2,1))"],
                setup_sweep_name=solution_full_name,
                domain="Sweep"
            )

            if sol_data:
                target_filename = f"{split}_sample_{global_id}.csv"
                full_export_path = os.path.join(export_dir, target_filename)

                sol_data.export_data_to_csv(full_export_path)

                with open(log_file_path, "a") as f:
                    f.write(
                        f"{target_filename},{split},{task['l1']},{task['l2']},{task['l3']},{task['g1']},{task['g2']}\n")

                try:
                    app.cleanup_solution()
                except Exception as cleanup_err:
                    print(f"[Worker {worker_id}] Warning: Failed to cleanup solution. ({str(cleanup_err)})")
            else:
                print(f"[Worker {worker_id}] Error: No solution data retrieved for Global #{global_id}")

            print(f"[Worker {worker_id}] Finished Global #{global_id}. Time: {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"\n[Worker {worker_id}] CRITICAL ERROR: {str(e)}")
        traceback.print_exc()

    finally:
        close_hfss_session()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 1. 仅在主程序启动时执行一次大扫除
    clean_environment(project_path)

    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project not found: {project_path}")

    # 2. 为每个 Worker 复制独立的工程文件
    worker_projects = []
    for i in range(NUM_WORKERS):
        new_proj = duplicate_project(project_path, i)
        worker_projects.append(new_proj)

    # 3. 构建总任务队列
    # train_range is 33.0,35.0,0.1
    # val_range is 33.25,35.0,0.1
    train_l_range = np.arange(33.25, 35.0, 0.1)
    train_g_range = np.arange(1.75, 4.0, 0.1)
    val_l_range = np.arange(33.25, 35.0, 0.1)
    val_g_range = np.arange(1.75, 4.0, 0.1)

    all_tasks = []
    global_counter = 0

    for _ in range(6300):
        l2_val = round(random.choice(train_l_range), 2)
        all_tasks.append({
            "global_id": global_counter, "split": "train",
            "l1": round(random.choice(train_l_range), 2), "l2": l2_val, "l3": l2_val,
            "g1": round(random.choice(train_g_range), 2), "g2": round(random.choice(train_g_range), 2)
        })
        global_counter += 1

    for _ in range(2000):
        l2_val = round(random.choice(val_l_range), 2)
        all_tasks.append({
            "global_id": global_counter, "split": "val",
            "l1": round(random.choice(val_l_range), 2), "l2": l2_val, "l3": l2_val,
            "g1": round(random.choice(val_g_range), 2), "g2": round(random.choice(val_g_range), 2)
        })
        global_counter += 1

    # 4. 将任务均匀分配给 3 个 Worker
    # 使用切片分配: worker0 拿 0,3,6... worker1 拿 1,4,7...
    worker_tasks = [all_tasks[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

    # 5. 启动多进程并发流水线
    print(f"\n[*] Starting Multiprocessing Pipeline with {NUM_WORKERS} workers...")
    start_time_all = time.time()

    processes = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=run_simulations_worker,
            args=(i, worker_tasks[i], worker_projects[i], export_folder)
        )
        processes.append(p)
        p.start()

    # 阻塞主进程，等待所有 Worker 执行完毕
    for p in processes:
        p.join()

    print(f"\n[*] All Parallel Tasks Completed! Total Time: {(time.time() - start_time_all) / 3600:.2f} hours.")