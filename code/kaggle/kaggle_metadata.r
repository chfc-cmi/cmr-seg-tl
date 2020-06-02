library(tidyverse)

dicom_metadata <- read_tsv("analysis/kaggle/dicom_metadata.tsv.xz", guess_max = 200000) %>% rename(Id=pid)
used_files <- read_tsv("analysis/kaggle/used_dicoms.log", col_names=c("pid","slice","frame","dir","file"), guess_max=200000) %>% mutate(Id=as.numeric(str_remove(pid,"_[ab]")), used = T)
metadata <- left_join(dicom_metadata,used_files,by=c("Id","dir","file")) %>%
    replace_na(list(used=F)) %>%
    mutate(set=if_else(`Patient ID`<501,'train',if_else(`Patient ID`>700,'test','validate')))

write_tsv(metadata, "analysis/kaggle/combined_metadata.tsv.xz")

patient_dim <- read_tsv("analysis/kaggle/patient_dimensions.tsv")
truth_train <- read_csv("analysis/kaggle/truth/train.csv") %>% mutate(EF = (Diastole-Systole)/Diastole, set="train")
truth_val <- read_csv("analysis/kaggle/truth/validate.csv") %>% mutate(EF = (Diastole-Systole)/Diastole, set="validate")
truth_test <- read_csv("analysis/kaggle/truth/solution.csv") %>%
    separate(Id, into=c("Id", "Phase"), sep="_") %>%
    spread(Phase, Volume) %>%
    select(-Usage) %>%
    mutate(Id = as.numeric(Id), EF = (Diastole-Systole)/Diastole, set="test")
truth <- bind_rows(truth_train, truth_val, truth_test)

patient_metadata <- metadata %>%
    select(
        PatientID=`Patient ID`,
        PatientsAge=`Patient\'s Age`,
        PatientsSex=`Patient\'s Sex`,
        pid,
        Columns,
        Rows,
        used,
        set,
        EchoNumber=`Echo Number(s)`,
        EchoTime=`Echo Time`,
        EchoTrainLength=`Echo Train Length`,
        FlipAngle=`Flip Angle`,
        PhaseEncodingDirection=`In-plane Phase Encoding Direction`,
        MagneticFieldStrength=`Magnetic Field Strength`,
        ModelName=`Manufacturer\'s Model Name`,
        NumberOfAverages=`Number of Averages`,
        #NumberOfFrames=`Number of Frames`,
        NumberOfPhaseEncodingSteps=`Number of Phase Encoding Steps`,
        PercentSampling=`Percent Sampling`,
        PixelBandwidth=`Pixel Bandwidth`,
        PixelRepresentation=`Pixel Representation`,
        PixelSpacing=`Pixel Spacing`,
        RepetitionTime=`Repetition Time`,
        ScanningSequence=`Scanning Sequence`,
        SequenceName=`Sequence Name`,
        SequenceVariant=`Sequence Variant`,
        SliceThickness=`Slice Thickness`, # we would need to use the empirical thickness
        SoftwareVersion=`Software Versions`,
        # SpacingBetweenSlices=`Spacing Between Slices`, # we would need to use the empirical thickness
        dBdt=`dB/dt`) %>%
    separate(PatientsAge, into=c("PatientsAgeNum","PatientsAgeUnit"), sep=-1) %>%
    mutate(
        PatientsAgeYears=as.numeric(PatientsAgeNum),
        PatientsAgeYears=if_else(
            PatientsAgeUnit=="Y",
            PatientsAgeYears,
            if_else(PatientsAgeUnit=="M",PatientsAgeYears/12,PatientsAgeYears/52)
        )
    ) %>%
    left_join(patient_dim, by="pid") %>%
    filter(used) %>%
    unique

# some are still there more than once... e.g. different EchoTime or RepetitionTime or ContrastAgent
patients_with_inconsistent_metadata <- patient_metadata %>%
    count(pid) %>%
    filter(n>1) %>%
    pull(pid)

# keep only one row per patient but remember if metadata was inconsistent
patient_metadata <- patient_metadata %>%
    group_by(pid) %>%
    filter(row_number()==1) %>%
    ungroup %>%
    mutate(inconsistentMetadata = pid %in% patients_with_inconsistent_metadata)

patient_metadata <- patient_metadata %>% left_join(truth, by=c("PatientID"="Id","set")) 

write_tsv(patient_metadata, "analysis/kaggle/patient_metadata.tsv")
