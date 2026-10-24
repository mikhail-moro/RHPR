-- Use StudyStore.connect() or rebuild study.duckdb views after moving files.
-- Explore validation first. Exact-panel averages require sampling weights.

SELECT m.run_id,m.scoring,m.search,m.beam_width,count(*) AS support,
       avg(m.ndcg10) AS ndcg10,avg(m.recall10) AS recall10
FROM case_metrics m JOIN cases c USING(case_id)
WHERE c.split='validation' AND m.epoch=20
GROUP BY ALL ORDER BY m.scoring,m.search,m.beam_width,m.run_id;

-- Same case, changed index: SID and first-level centroid margin.
SELECT p.*,c.user_id,c.target_item,c.target_repeats,
       a.sid AS sid_a,b.sid AS sid_b,
       ga.distance_margin AS margin_a,gb.distance_margin AS margin_b
FROM paired_quality p JOIN cases c USING(case_id)
JOIN item_codes a ON a.run_id=p.run_a AND a.item_id=c.target_item
JOIN item_codes b ON b.run_id=p.run_b AND b.item_id=c.target_item
JOIN geometry ga ON ga.run_id=p.run_a AND ga.item_id=c.target_item AND ga.level=0
JOIN geometry gb ON gb.run_id=p.run_b AND gb.item_id=c.target_item AND gb.level=0
WHERE c.split='validation' AND p.epoch=20 AND p.search='beam'
ORDER BY abs(p.delta_ndcg10) DESC LIMIT 100;

-- Keep undefined item quality visible; replace123 with an actual item ID.
SELECT i.item_id,i.text,i.train_events,q.*,s.sid
FROM items i JOIN item_codes s USING(item_id)
LEFT JOIN item_quality q ON q.item_id=i.item_id AND q.run_id=s.run_id
     AND q.split='validation' AND q.epoch=20 AND q.search='beam'
WHERE i.item_id=123;

-- Complete history versus actual context, linked to every index realization.
SELECT c.case_id,c.target_item,h.position,h.visible,h.input_position,
       h.item_id,i.text,s.run_id,s.sid
FROM cases c JOIN case_history h USING(case_id)
JOIN items i ON i.item_id=h.item_id JOIN item_codes s ON s.item_id=h.item_id
WHERE c.case_id='validation:123:8' ORDER BY s.run_id,h.position;

-- Raw probability mass allocated to valid catalogue items, panel-weighted.
SELECT m.run_id,count(*) AS panel_support,
       sum(m.valid_catalogue_mass/c.panel_probability)
       /sum(1/c.panel_probability) AS weighted_valid_mass
FROM case_metrics m JOIN cases c USING(case_id)
WHERE c.split='validation' AND m.epoch=20 AND m.search='exact' AND m.scoring='raw'
GROUP BY m.run_id;

-- Search misses and lucky beam promotions under MATCHING scoring.
SELECT m.run_id,m.scoring,m.beam_width,count(*) AS panel_cases,
       avg(m.exact_top10_overlap) AS top10_overlap,
       count(*) FILTER(WHERE m.target_lost10) AS lost_targets,
       count(*) FILTER(WHERE m.target_promoted10) AS promoted_targets
FROM case_metrics m JOIN cases c USING(case_id)
WHERE c.split='validation' AND m.epoch=20 AND m.search='beam'
      AND m.exact_top10_overlap IS NOT NULL GROUP BY ALL;

-- Actual training supervision/loss for token12 at level0, per realization.
-- Numeric code labels DO NOT identify the same group across different runs.
SELECT e.run_id,e.epoch,sum(e.target_count) AS targets,
       sum(list_extract(e.target_level_loss_sums,1))/nullif(sum(e.target_count),0) AS token_loss
FROM exposures e JOIN item_codes s USING(run_id,item_id)
WHERE list_extract(s.sid,1)=12 GROUP BY ALL ORDER BY e.run_id,e.epoch;

-- A global vocabulary token's reuse under different trie parents.
SELECT run_id,level,token,items,parent_count,train_events
FROM token_usage WHERE level=1 ORDER BY parent_count DESC LIMIT 100;
