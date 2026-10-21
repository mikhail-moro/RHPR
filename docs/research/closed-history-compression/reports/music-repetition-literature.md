> Historical report from 2026-10-20; research branch closed on 2026-10-21. Any proposed runs or next steps below are historical and cancelled. See the [closure record](../README.md) for the final status.

# Музыкальные повторы, контекст и следующий исследовательский вопрос

Проверено по первичным источникам 20 октября 2026. Это обзор прочитанных работ, а не систематический обзор всей литературы и не доказательство отсутствия аналогов предлагаемой узкой идеи.

Сильная персональная частота в наших данных правдоподобна для музыки. Из этого не следует ни ошибка validation, ни некорректность всякого next-item prediction, ни научная новизна самого учета повторов. Требуется выбрать реальный сценарий и сопоставить objective, observable labels и оценку.

## Восемь источников, которые непосредственно меняют исследовательский план

1. **Abbattista et al., “Enhancing Sequential Music Recommendation with Personalized Popularity Awareness”, RecSys 2024.** [Полный текст](https://arxiv.org/html/2409.04329v1), [код авторов](https://github.com/sisinflab/personalized-popularity-awareness).

   Personalized Most Popular превосходит vanilla SASRec, gSASRec и BERT4Rec в их музыкальных экспериментах. Авторы добавляют персональный частотный prior к neural scores, чтобы модель учила отклонение от него; для causal SASRec counts используют только соответствующий прошлый prefix. Это прямой prior art к нашему неожиданному CPU результату и к идее «оставить повторы простому механизму». Ограничения: вход 150, popularity-sampled каталог 30k, Last.fm-1K и Yandex Music Event 2019; NDCG по агрегированным релевантностям held-out периода, а не наш одиночный first-event-of-day target. Их значения NDCG и процент улучшения нельзя напрямую переносить к нам. Работу нельзя списать на отсутствие поиска гиперпараметров: авторы сообщают большой HPO-бюджет.

2. **Tran, Salha-Galvan, Sguerra, Hennequin, “Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation”, RecSys 2024 (PISA, Deezer).** [Полный текст](https://arxiv.org/html/2408.16578v1), [код и публичный Deezer dataset](https://github.com/deezer/recsys24-pisa).

   Предсказывается множество песен следующей сессии по предыдущим сессиям. Сессии разделяются ≥20 минутами неактивности; сохраняются первые 10 треков. ACT-R frequency/recency участвует в session embedding и долгосрочной памяти; Transformer моделирует последовательность сессий. Отдельно оцениваются repeated/explored targets, повторность рекомендаций и её смещение. Ограничения: fixed first10 и специфическая фильтрация популярных треков/активных пользователей; до 20/30 предшествующих сессий; embedding/preprocessing отличаются от нашего каталога. Session-set target не является прямым измерением настроения, оптимальной подборки или удовлетворенности. Сам переход с NIP на next-session set и explicit repetition уже исследован.

3. **Reiter-Haas et al., “Predicting Music Relistening Behavior Using the ACT-R Framework”, RecSys 2021.** [Полный текст](https://arxiv.org/html/2108.02138v1), [код](https://github.com/socialcomplab/recsys21-relistening-actr).

   На Last.fm исследуются recency/frequency, co-occurrence, сходство и familiarity при музыкальном relistening. Base-level memory activation — сильный отдельный компонент; сочетания компонентов улучшают R-precision. В анализе распределения повторов отмечены временная зависимость и периодические пики, включая сутки. Важное ограничение: это прежде всего предсказание повторного прослушивания, с собственными session/candidate/evaluation условиями; это не доказательство оптимального баланса discovery и familiar tracks. Нашим минимальным baseline должен стать time-decayed personal frequency / BLL, а не только lifetime frequency и neural recent.

4. **Ren et al., “RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-Based Recommendation”, AAAI 2019.** [Страница конференции](https://ojs.aaai.org/index.php/AAAI/article/view/4408), [препринт](https://arxiv.org/abs/1812.02646).

   Нейросетевой repeat–explore механизм выбирает между повтором item из истории и рекомендованием других items. Это established prior art для отдельной repeat-памяти, copy-like декодера и переключения режимов. Музыка — один из мотивирующих доменов, но abstract не устанавливает superiority на нашем датасете/протоколе. Нельзя заявлять новую идею как «обучим gate повторять или исследовать»: новизна должна лежать в конкретной постановке, информации, поведении, стоимости или обоснованном изменении механизма.

5. **Mallamaci et al., “Balancing Accuracy and Novelty with Sub-Item Popularity”, RecSys 2025.** [Полный текст](https://arxiv.org/html/2508.05198v1), [код](https://github.com/sisinflab/Sub-id-Popularity).

   Персональную частоту считают не только по exact track ID, но и по latent sub-IDs RecJPQ. Это позволяет усиливать familiar latent structure, рекомендуя другой track, и регулировать accuracy/novelty через score mixing. Это близкий аналог интуиции «повторять не сам трек, а знакомый стиль/набор свойств». Ограничения: latent sub-ID не является валидированным genre/mood/artist label; работа в основном демонстрирует offline trade-off на Last.fm и Yandex Event с каталогом 30k. Новизна по frequency-based метрике не доказывает субъективную discovery utility и длительное удовлетворение. Для нас это обязательный related work при любой идее об абстрактных музыкальных повторениях.

6. **Sguerra, Tran, Hennequin, Moussallam, “Uncertainty in Repeated Implicit Feedback as a Measure of Reliability”, UMAP 2025 (Deezer).** [Полный текст](https://arxiv.org/html/2505.02492v1).

   Рассматривается ненадежность предпосылки «больше повторов → увереннее предпочтение»: exposure, satiation и давность меняют смысл play counts. Предлагается Bayesian confidence/reliability для implicit listening feedback. Публично описанный набор содержит около 11 млн interactions и skip/positive labels; крайне частые user–track пары отфильтрованы. Это полезная постановка для оценки того, когда частота должна считаться сильным свидетельством. Ограничения: конкретные фильтры и feedback labels; сырые listen-only IDs не позволяют валидировать dislike/satiation или причинное влияние следующей рекомендации. Даже частое воспроизведение может иметь разные функции для пользователя.

7. **Marey, Sguerra, Moussallam, “Modeling Activity-Driven Music Listening with PACE”, CHIIR 2024 (Deezer).** [Полный текст](https://arxiv.org/html/2405.01417v1), [код](https://github.com/deezer/modeling_activity_pace).

   Долгосрочные недельные паттерны слушания кодируются как multichannel user time series; embeddings проверяются на предсказании self-reported activities (работа, спорт, транспорт и т.д.). Каналы включают объем, repetitiveness, organic/algorithmic origin и liked-content usage. Авторы отличают soundtracking от declared general taste. Ограничения важны: survey сообщает общие практики пользователя, а не истинную activity/mood каждого timestamp; нужны дополнительные каналы и ответы опроса. Из одних item IDs и времени нельзя честно назвать cluster «настроением» или доказать, что каждый возврат — ритуал. Нам доступны observable recurrence patterns, а semantic interpretation требует данных/валидации.

8. **Manolovitz, Ogihara, “Practical Evaluation of Repeated Recommendations in Personalized Music Discovery”, ISMIR 2020.** [PDF proceedings](https://program.ismir2020.net/static/final_papers/13.pdf).

   Небольшой шестинедельный эксперимент с 19 Spotify users и follow-up через месяц проверяет, повышает ли повторное предъявление новой песни вероятность добровольного возвращения к ней. Предлагается retention rate обнаруженной музыки как дополнение к liking. Авторы находят связь repeated exposure и revisiting. Ограничения: малая выборка, playlists/surveys и конкретный дизайн; отсутствует универсальная гарантия «больше повторов всегда лучше». Содержательный music-specific вопрос — превращается ли discovery в новую любимую музыку, а не только попал ли точный следующий track. Для причинного вывода о частоте рекомендаций нужны соответствующий эксперимент или exposure logs.

## Что уже известно и не составляет достаточную новизну

- Повторы — нормальная часть музыкального потребления и сильный предиктивный сигнал.
- Personalized popularity, time decay / BLL / ACT-R, repeat–explore mixtures и session-set prediction имеют прямой prior art.
- Музыкальный интерес ситуативен: activity/background listening не обязаны совпадать с глобальным taste profile.
- Поддержка discovery и получение familiar-content satisfaction — разные задачи; exact next-track hit не измеряет обе полностью.
- «Сделаем mood-aware модель» без надежного mood label и наблюдаемого decision context — недостаточно конкретная проверяемая постановка.

## Реальные узкие вопросы для следующего исследования

Это авторская исследовательская интерпретация прочитанного, **не утверждение доказанной мировой новизны**.

1. **Что длинная память добавляет сверх recency/frequency?** Отдельно восстановление давних любимых треков, текущий знакомый repertoire и discovery. Сначала personal frequency, time-decayed frequency/BLL, simple transition/session-neighbor controls; затем incremental gains и стоимость. Механизм долгой памяти имеет смысл, если улучшается задача, где короткий repeat prior объективно недостаточен.
2. **Зависит ли ранжирование методов от горизонта и определения события?** Exact next-track, первые K songs после session gap, set следующих K songs, session-start targets. Анализировать одинаковые temporal boundaries/catalogue, указывать произвольность gap threshold и размер target set. Это протокольный вопрос, а не доказательство, что один objective «правильный всегда».
3. **Возвращение к repertoire против возвращения к exact ID.** Повтор exact track, known artist/album, familiar embedding cluster with new track, truly unfamiliar area. Сравнить hierarchical repeat baselines. Для artist/album необходима metadata; embedding cluster обозначать cluster, не mood. Latent-level popularity уже покрыта sPPS2025, поэтому одной замены ID на cluster недостаточно.
4. **Сколько истории нужно для возобновления старого интереса после паузы?** Re-entry / resurfacing старого repertoire после длительного gap является наблюдаемым target; тестировать отложенные любимые tracks/sessions сверх сильного time-decay prior. Нужны прошлые timestamps, честная censoring/history-window обработка и past-only context features.
5. **Как distinguish familiarity, fatigue и discovery adoption?** С listen-only данными это ограниченно идентифицируемо. Для skip-aware reliability пригодны richer feedback datasets; для adoption — добровольные later returns/likes и достаточно длинное follow-up. Повторы после того же autoplay playlist не тождественны новой удовлетворенности. Предсказывать интерес и доказывать причинный эффект рекомендации — разные уровни доказательств.

В описании наших срезов использовать «не встречался в доступном H+R», а не автоматически «новый пользователю». Возвращение из-за очень старой истории может выглядеть как discovery при коротком окне. Обобщения из сверхактивного long-history cohort на весь музыкальный сервис требуют отдельной проверки.
