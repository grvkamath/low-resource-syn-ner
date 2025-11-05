library(tidyverse)

finetuning <- read_csv("./results/finetuning_synth_eval.csv")
fromscratch <- read_csv("./results/fromscratch_synth_eval.csv")

finetuning_plot <- finetuning %>% 
  filter(raw_data_source!='pre_training') %>% 
  mutate(data_source = if_else(data_source %in% c("MasakhaNER 2", "Naamapadam", "Universal NER"), 'Organic', data_source)) %>%
  mutate(Language = case_when(
    lang == "da"  ~ "Danish",
    lang == "ibo" ~ "Igbo",
    lang == "kin" ~ "Kinyarwanda",
    lang == "kn"  ~ "Kannada",
    lang == "ml"  ~ "Malayalam",
    lang == "sk"  ~ "Slovak",
    lang == "sv"  ~ "Swedish",
    lang == "swa" ~ "Swahili",
    lang == "ta"  ~ "Tamil",
    lang == "te"  ~ "Telugu",
    lang == "yor" ~ "Yoruba",
    TRUE ~ lang
  )) %>%
  ggplot(aes(x=size, y=f1, color=data_source)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Fine-tuning at different sizes after prior NER training on related language",
    x = "Fine-tuning training data size",
    y = "F1 Score",
    color = "Data Source"
  ) +
  ylim(0, 1) +
  xlim(0, 5000) +
  facet_wrap(~ Language) +
  # Set different text sizes for the facet labels and the main title
  theme(text = element_text(size = 16),
        strip.text = element_text(size = 14),
        title = element_text(size = 18))

ggsave("finetuning_synth_eval.png", finetuning_plot, width = 16, height = 8)

fromscratch_plot <- fromscratch %>%
  filter(raw_data_source!='pre_training') %>%
  mutate(data_source = if_else(data_source %in% c("MasakhaNER 2", "Naamapadam", "Universal NER"), 'Organic', data_source)) %>%
  mutate(Language = case_when(
    lang == "da"  ~ "Danish",
    lang == "ibo" ~ "Igbo",
    lang == "kin" ~ "Kinyarwanda",
    lang == "kn"  ~ "Kannada",
    lang == "ml"  ~ "Malayalam",
    lang == "sk"  ~ "Slovak",
    lang == "sv"  ~ "Swedish",
    lang == "swa" ~ "Swahili",
    lang == "ta"  ~ "Tamil",
    lang == "te"  ~ "Telugu",
    lang == "yor" ~ "Yoruba",
    TRUE ~ lang
  )) %>%
  ggplot(aes(x=size, y=f1, color=data_source)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Fine-tuning at different sizes with no prior NER training",
    x = "Fine-tuning training data size",
    y = "F1 Score",
    color = "Data Source"
  ) +
  facet_wrap(~ Language) +
  ylim(0, 1) +
  xlim(0, 5000) + 
  theme(text = element_text(size = 16),
        strip.text = element_text(size = 14),
        title = element_text(size = 18))

ggsave("fromscratch_synth_eval.png", fromscratch_plot, width = 16, height = 8)

