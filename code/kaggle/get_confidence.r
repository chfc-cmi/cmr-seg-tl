library(tidyverse)

ukbb_pred <- read_csv("ukbb_ventricular_volumes.csv") %>% rename(Id = X1) %>% mutate(pid=str_replace_all(Id,"_rot90",""))
metadata <- read_tsv("patient_metadata.tsv")
all_images <- read_tsv("image_list.tsv.xz", col_names = c("Id", "file"), col_types = "cc")

data <- ukbb_pred %>% select(Id,pid,LVEDV=`LVEDV (mL)`,LVESV=`LVESV (mL)`,LVEF=`LVEF (%)`) %>%
    left_join(metadata, by="pid")

#data <- data %>% mutate(AgeBin = cut(PatientsAgeYears, breaks=c(0,20,40,70,100)))

diff_data <- data %>%
    mutate(sys_diff = LVESV-Systole, sys_perc = sys_diff/Systole) %>%
    mutate(dia_diff = LVEDV-Diastole, dia_perc = dia_diff/Diastole) %>%
    mutate(max_abs_perc_diff = pmax(abs(sys_perc),abs(dia_perc)))

# if there is a rotated version, keep only that one
all_images %>%
    left_join(select(diff_data,Id,pid,score=max_abs_perc_diff,set)) %>%
    mutate(rot90=str_detect(Id,"_rot90")) %>%
    group_by(pid) %>%
    mutate(rot_count=sum(rot90)) %>%
    filter(rot_count==0 | rot90) %>% 
    ungroup %>%
    select(-rot90,-rot_count) %>%
    mutate(is_val= (set=="validate")) %>%
    write_tsv("image_list_filtered_score.tsv")

