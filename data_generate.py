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
# 0. simulation configuration
# ==========================================
NUM_WORKERS = 3  
CORES_PER_TASK = 16  # 3 process will occupy 48 core
SKIP_VALIDATION = True

# enviroment configuration
os.environ["ANSYSEM_ROOT241"] = "/tools/ansys/ansysem/2024R1/v241/Linux64/"
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["ANSYSEM_ROOT241"]

# Targer project path
project_dir = ""
project_name = "Project10"
project_path = os.path.join(project_dir, f"{project_name}.aedt")
export_folder = os.path.join(project_dir, "dataset_surrogate_model_523")

if not os.path.exists(export_folder):
    os.makedirs(export_folder)


# ==========================================
# prepare for your enviroment
# ==========================================
def clean_environment(proj_path):
    """At the begining of the program"""
    current_user = getpass.getuser()
    print(f"[*] Cleaning up the Ansys processes of  {current_user}...")

    os.system(f"pkill -9 -u {current_user} -f ansysedt > /dev/null 2>&1")
    os.system(f"pkill -9 -u {current_user} -f ansoftdxe > /dev/null 2>&1")
    time.sleep(3) 

    lock_file = proj_path + ".lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception:
            pass


def duplicate_project(original_path, worker_id):
    base_dir = os.path.dirname(original_path)
    base_name = os.path.splitext(os.path.basename(original_path))[0]

    new_name = f"{base_name}_worker{worker_id}"
    new_proj_path = os.path.join(base_dir, f"{new_name}.aedt")

    shutil.copy2(original_path, new_proj_path)

    original_aedb = os.path.join(base_dir, f"{base_name}.aedb")
    new_aedb = os.path.join(base_dir, f"{new_name}.aedb")
    if os.path.exists(original_aedb):
        if os.path.exists(new_aedb):
            shutil.rmtree(new_aedb)
        shutil.copytree(original_aedb, new_aedb)

    return new_proj_path


# ==========================================
# main function
# ==========================================
def run_simulations_worker(worker_id, tasks, proj_path, export_dir):
    log_file_path = os.path.join(export_dir, f"final_data_log_worker{worker_id}.csv")

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("filename,split,l1,l2,l3,g1,g2\n")

    total_samples = len(tasks)
    print(f"[Worker {worker_id}] begin.the number: {total_samples}")

    RESTART_THRESHOLD = 150
    app = None
    desktop = None

    def start_hfss_session():
        nonlocal app, desktop
        print(f"\n[Worker {worker_id}] Desktop initiation...")

        # new_desktop_session=True
        desktop = Desktop(
            version="2024.1",
            non_graphical=True,
            new_desktop_session=True
        )
        time.sleep(5)

        app = Hfss(project=proj_path, version="2024.1")

        if getattr(app, "oproject", None) is None:
            raise RuntimeError(f"[Worker {worker_id}] HFSS process load error！")

        if not app.design_list:
            raise RuntimeError(f"[Worker {worker_id}] can't find any Design！")

        app.set_active_design(app.design_list[0])

        setup_name = app.setup_names[0] if app.setup_names else None
        sweep_name = app.get_sweeps(setup_name)[0] if setup_name and app.get_sweeps(setup_name) else None

        if not setup_name or not sweep_name:
            raise RuntimeError(f"[Worker {worker_id}] can't find Setup or Sweep.")

        print(f"[Worker {worker_id}] successfully begin！")
        return setup_name, sweep_name

    def close_hfss_session():
        nonlocal app, desktop
        if app is not None or desktop is not None:
            print(f"[Worker {worker_id}] release memory")
            try:
                if app is not None:
                    app.release_desktop(close_projects=True, close_desktop=True)
            except Exception as e:
                print(f"[Worker {worker_id}] warning: {str(e)}")
            finally:
                app = None
                desktop = None
            time.sleep(5)

    try:
        active_setup, active_sweep = start_hfss_session()
        solution_full_name = f"{active_setup} : {active_sweep}"

        for i, task in enumerate(tasks):
            if i > 0 and i % RESTART_THRESHOLD == 0:
                print(f"\n[Worker {worker_id}] finish {RESTART_THRESHOLD} calculation，restarting...")
                close_hfss_session()
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
# main process
# ==========================================
if __name__ == '__main__':
    clean_environment(project_path)

    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project not found: {project_path}")

    worker_projects = []
    for i in range(NUM_WORKERS):
        new_proj = duplicate_project(project_path, i)
        worker_projects.append(new_proj)

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
.
    worker_tasks = [all_tasks[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

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

    for p in processes:
        p.join()

    print(f"\n[*] All Parallel Tasks Completed! Total Time: {(time.time() - start_time_all) / 3600:.2f} hours.")
