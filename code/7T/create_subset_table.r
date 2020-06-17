library(tidyverse)

train_files <- fs::dir_ls("data/7T/images/train") %>%
    enframe %>%
    transmute(
        file=str_replace(value,".*train/",""),
        Id=str_replace(file,"_slice.*",""),
        frame=str_replace(file,".*frame",""),
        frame=str_replace(frame,"-.*",""),
        frame=as.numeric(frame),
        slice=str_replace(file,".*slice",""),
        slice=str_replace(slice,"_frame.*",""),
        slice=as.numeric(slice)
    )

set.seed(42)

volunteers_7 <- train_files %>% pull(Id) %>% unique %>% sample(7,replace=FALSE)
volunteers_3 <- volunteers_7 %>% sample(3,replace=FALSE)
volunteers_1 <- volunteers_3 %>% sample(1,replace=FALSE)
print(volunteers_7)
print(volunteers_3)
print(volunteers_1)

train_files <- train_files %>% mutate(
    v7 = Id %in% volunteers_7,
    v3 = Id %in% volunteers_3,
    v1 = Id %in% volunteers_1
)

file_random_order <- train_files %>% pull(file) %>% sample

train_files <- train_files %>% mutate(
    r7 = file %in% head(file_random_order, sum(v7)),
    r3 = file %in% head(file_random_order, sum(v3)),
    r1 = file %in% head(file_random_order, sum(v1))
)

# Add subsets with end-systolic and end-diastolic frames

pixel_counts <- read_tsv("analysis/7T/pixel_counts_by_class.tsv")

pixel_counts_by_tf <- pixel_counts %>%
    filter(source == "7T") %>%
    separate(file, "_", into=c("Id","slice","frame")) %>%
    mutate(
        slice=str_replace(slice,"slice",""),
        slice=as.numeric(slice),
        frame=str_replace(frame,"frame",""),
        frame=str_replace(frame,"-mask.png",""),
        frame=as.numeric(frame)
    ) %>%
    group_by(Id, frame) %>%
    summarize(bg=sum(bg),lv=sum(lv),my=sum(my))

es_frame <- pixel_counts_by_tf %>% group_by(Id) %>% top_n(1,-lv) %>% ungroup %>% transmute(id_frame=paste(Id, frame, sep="_")) %>% pull
ed_frame <- pixel_counts_by_tf %>% group_by(Id) %>% top_n(1,lv) %>% ungroup %>% transmute(id_frame=paste(Id, frame, sep="_")) %>% pull
es_tbl <- pixel_counts_by_tf %>% group_by(Id) %>% top_n(1,-lv) %>% ungroup %>% select(Id, es=frame)
ed_tbl <- pixel_counts_by_tf %>% group_by(Id) %>% top_n(1,lv) %>% ungroup %>% select(Id, ed=frame)

left_join(es_tbl, ed_tbl, by="Id") %>% write_tsv("analysis/7T/esed_frames.tsv")

train_files <- train_files %>% mutate(
    esed = paste(Id,frame,sep="_") %in% c(es_frame,ed_frame),
    r_esed = file %in% head(file_random_order, sum(esed)),
)

train_files %>% write_tsv("analysis/7T/image_subsets.tsv")

train_files %>% summarise(all=n(),sum(v7),sum(v3),sum(v1),sum(r7),sum(r3),sum(r1),sum(esed),sum(r_esed))