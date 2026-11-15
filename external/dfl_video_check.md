# DFL sync: video check sheet

Checks whether the snapped event times match what really happened, using public match video. The sync method was frozen (commit b79114c) before this sheet was made, so this only measures it. Nothing gets tuned on the result.

## How to use it

1. Open the match's highlights: each match below links where they were found. Those German pages may be blocked outside Germany; if so, search YouTube for the two teams plus "highlights" and the season. Each goal below gives the game clock of the shot; highlights show it on screen.
2. For each goal, step through the build-up frame by frame (VLC: press E) and time the gap between each listed touch and the one before it. Write it in **Gap, your timing**.
3. Skip a touch if the clip doesn't show it, or write `?`.

We compare your gaps with the **raw log** and **snapped** columns. Snapping is working if the snapped gaps are clearly closer to yours, typically within about 0.2 s.

Notes:

* Clock times come from the tracking (seconds since that half's kick-off), so they may differ from the broadcast clock by a second or so. Use them to find the moment, not to check it.
* Second-half times run from 45:00. Stoppage time shows as 45:xx or 90:xx and beyond, where the broadcast shows 45+x or 90+x.
* A touch marked "no (lag only)" couldn't be snapped and keeps the lag-corrected log time; its gaps are expected to be worse.
* Penalties and own goals may be logged differently; if a goal from the final score is missing below, that's why.

## 1. FC Köln – FC Bayern München (Bundesliga, 2023-05-27, 1:2)

Match J03WMX. Logged goals: 3; final score 1:2.

Highlights: <https://www.zdf.de/video/kurzfassungen/bundesliga-saison-2022-2023-fussball-highlights-100/1-fc-koeln-bayern-muenchen-highlights-zusammenfassung-100> · <https://fcbayern.com/en/matches/profis/bundesliga/2022-2023/1-fc-koeln-fc-bayern-muenchen-27-05-2023/videos>

### Goal 1: Kingsley Coman (FC Bayern München), shot at about 07:48.2 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Ryan Gravenberch | pass | 07:37.6 | – | – |  | yes |
| 2 | T. Müller | pass | 07:39.3 | 1.72 s | 1.72 s |  | yes |
| 3 | Leroy Sané | pass | 07:44.8 | 4.80 s | 5.44 s |  | yes |
| 4 | Kingsley Coman | shot | 07:48.2 | 5.32 s | 3.48 s |  | yes |

### Goal 2: D. Ljubicic (1. FC Köln), shot at about 80:04.0 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | D. Ljubicic | Penalty | 80:04.0 | – | – |  | yes |

### Goal 3: J. Musiala (FC Bayern München), shot at about 88:14.2 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Joshua Kimmich | pass | 88:05.2 | – | – |  | yes |
| 2 | Joao Cancelo | pass | 88:08.4 | 1.88 s | 3.16 s |  | yes |
| 3 | Serge Gnabry | pass | 88:10.0 | 3.48 s | 1.60 s |  | yes |
| 4 | J. Musiala | shot | 88:14.2 | 3.52 s | 4.28 s |  | yes |

## VfL Bochum 1848 – Bayer 04 Leverkusen (Bundesliga, 2023-05-27, 3:0)

Match J03WN1. Logged goals: 3; final score 3:0.

Highlights: <https://www.zdf.de/sport/bundesliga/vfl-bochum-bayer-leverkusen-highlights-zusammenfassung-100.html> · <https://www.espn.com/soccer/match/_/gameId/638742>

### Goal 1: P. Förster (VfL Bochum 1848), shot at about 18:03.2 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | P. Förster | pass | 17:52.2 | – | – |  | yes |
| 2 | K. Stöger | pass | 17:54.6 | 1.80 s | 2.40 s |  | yes |
| 3 | T. Asano | cross | 18:01.6 | 5.28 s | 7.00 s |  | yes |
| 4 | P. Förster | shot | 18:03.2 | 3.44 s | 1.56 s |  | yes |

### Goal 2: T. Asano (VfL Bochum 1848), shot at about 33:37.2 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | K. Stöger | CornerKick | 33:35.0 | – | – |  | yes |
| 2 | T. Asano | shot | 33:37.2 | 5.12 s | 2.12 s |  | yes |

### Goal 3: K. Stöger (VfL Bochum 1848), shot at about 85:18.7 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | M. Broschinski | keeper/claim | 85:14.8 | – | – |  | yes |
| 2 | M. Broschinski | pass | 85:15.8 | 0.00 s | 0.96 s |  | yes |
| 3 | S. Zoller | pass | 85:16.5 | 1.44 s | 0.76 s |  | yes |
| 4 | K. Stöger | shot | 85:18.7 | 3.40 s | 2.16 s |  | yes |

## Fortuna Düsseldorf – SSV Jahn Regensburg (2. Bundesliga, 2022-08-26, 4:0)

Match J03WOH. Logged goals: 4; final score 4:0.

Highlights: <https://www.bundesliga.com/es/2bundesliga/partidos/2022-2023/6/fortuna-duesseldorf-vs-ssv-jahn-regensburg/liveticker>

### Goal 1: D. Ginczek (Fortuna Düsseldorf), shot at about 58:57.0 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Jorrit Hendrix | pass | 58:48.7 | – | – |  | yes |
| 2 | Kristoffer Peterson | pass | 58:52.6 | 3.20 s | 3.92 s |  | yes |
| 3 | Dawid Kownacki | pass | 58:53.5 | 2.52 s | 0.88 s |  | yes |
| 4 | D. Ginczek | shot | 58:57.0 | 3.04 s | 3.48 s |  | yes |

### Goal 2: Dawid Kownacki (Fortuna Düsseldorf), shot at about 64:18.2 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Dawid Kownacki | Penalty | 64:18.2 | – | – |  | yes |

### Goal 3: F. Klaus (Fortuna Düsseldorf), shot at about 77:01.1 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Scott Kennedy | touch | 76:59.3 | – | – |  | no (lag only) |
| 2 | D. Ginczek | pass | 77:00.2 | 0.52 s | 0.88 s |  | yes |
| 3 | F. Klaus | pass | 77:01.1 | 0.44 s | 0.88 s |  | yes |
| 4 | F. Klaus | shot | 77:01.1 | 1.20 s | 0.04 s |  | yes |

### Goal 4: Shinta Appelkamp (Fortuna Düsseldorf), shot at about 84:05.5 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Kwadwo Baah | pass | 84:02.7 | – | – |  | yes |
| 2 | Scott Kennedy | touch | 84:02.7 | 1.96 s | 0.04 s |  | yes |
| 3 | K. Faber | touch | 84:04.4 | 0.16 s | 1.64 s |  | yes |
| 4 | Shinta Appelkamp | shot | 84:05.5 | 0.88 s | 1.12 s |  | yes |

## Fortuna Düsseldorf – F.C. Hansa Rostock (2. Bundesliga, 2022-09-10, 3:1)

Match J03WOY. Logged goals: 4; final score 3:1.

Highlights: <https://www.bundesliga.com/en/2bundesliga/matchday/2022-2023/8/fortuna-duesseldorf-vs-fc-hansa-rostock/liveticker>

### Goal 1: Dawid Kownacki (Fortuna Düsseldorf), shot at about 09:39.5 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | M. Karbownik | pass | 09:28.6 | – | – |  | yes |
| 2 | C. Klarer | pass | 09:31.7 | 4.88 s | 3.08 s |  | yes |
| 3 | Tim Oberdorf | pass | 09:37.2 | 5.44 s | 5.56 s |  | yes |
| 4 | Dawid Kownacki | shot | 09:39.5 | 3.60 s | 2.24 s |  | yes |

### Goal 2: M. Sobottka (Fortuna Düsseldorf), shot at about 26:42.0 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | M. Sobottka | keeper/claim | 26:38.9 | – | – |  | yes |
| 2 | M. Sobottka | pass | 26:38.9 | 0.00 s | 0.04 s |  | yes |
| 3 | Dawid Kownacki | pass | 26:39.6 | 1.84 s | 0.64 s |  | yes |
| 4 | M. Sobottka | shot | 26:42.0 | 2.96 s | 2.48 s |  | yes |

### Goal 3: Lukas Fröde (F.C. Hansa Rostock), shot at about 60:02.2 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Frederic Ananou | pass | 59:52.4 | – | – |  | yes |
| 2 | N. Neidhart | pass | 59:58.4 | 5.96 s | 6.00 s |  | yes |
| 3 | K. Pröger | cross | 60:00.8 | 4.16 s | 2.40 s |  | yes |
| 4 | Lukas Fröde | shot | 60:02.2 | 2.24 s | 1.36 s |  | yes |

### Goal 4: E. Iyoha (Fortuna Düsseldorf), shot at about 76:51.5 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | R. van Drongelen | pass | 76:45.9 | – | – |  | yes |
| 2 | M. Sobottka | pass | 76:46.7 | 2.52 s | 0.84 s |  | yes |
| 3 | A. Tanaka | pass | 76:48.3 | 1.44 s | 1.56 s |  | yes |
| 4 | E. Iyoha | shot | 76:51.5 | 2.96 s | 3.20 s |  | yes |

## Fortuna Düsseldorf – 1. FC Nürnberg (2. Bundesliga, 2022-10-15, 0:1)

Match J03WPY. Logged goals: 1; final score 0:1.

Highlights: <https://www.bundesliga.com/en/2bundesliga/matchday/2022-2023/12/fortuna-duesseldorf-vs-1-fc-nuernberg/liveticker>

### Goal 1: Kwadwo Duah (1. FC Nürnberg), shot at about 46:23.2 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Tim Oberdorf | touch | 46:18.4 | – | – |  | yes |
| 2 | L. Tempelmann | pass | 46:20.0 | 2.28 s | 1.52 s |  | yes |
| 3 | J. Castrop | pass | 46:21.7 | 1.52 s | 1.72 s |  | yes |
| 4 | Kwadwo Duah | shot | 46:23.2 | 2.16 s | 1.48 s |  | yes |

## Fortuna Düsseldorf – FC St. Pauli (2. Bundesliga, 2022-11-05, 1:0)

Match J03WQQ. Logged goals: 1; final score 1:0.

Highlights: <https://www.bundesliga.com/fr/2bundesliga/journee/2022-2023/15/fortuna-duesseldorf-vs-fc-st-pauli/liveticker>

### Goal 1: R. Hennings (Fortuna Düsseldorf), shot at about 21:02.4 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | F. Kastenmeier | GoalKick | 20:55.6 | – | – |  | no (lag only) |
| 2 | Dawid Kownacki | pass | 21:00.1 | 5.20 s | 4.52 s |  | yes |
| 3 | R. Hennings | shot | 21:02.4 | 3.00 s | 2.28 s |  | yes |

## Fortuna Düsseldorf – 1. FC Kaiserslautern (2. Bundesliga, 2022-11-11, 1:2)

Match J03WR9. Logged goals: 3; final score 1:2.

Highlights: <https://www.bundesliga.com/en/2bundesliga/news/fortuna-dusseldorf-kaiserslautern-live-team-news-match-stats-tanaka-boyd-durm-21877>

### Goal 1: M. Karbownik (Fortuna Düsseldorf), shot at about 13:13.6 on the clock (half 1)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | Kenny Prince Redondo | touch | 13:06.9 | – | – |  | yes |
| 2 | C. Klarer | touch | 13:07.5 | 0.28 s | 0.60 s |  | yes |
| 3 | E. Iyoha | pass | 13:08.6 | 1.12 s | 1.12 s |  | yes |
| 4 | M. Karbownik | shot | 13:13.6 | 4.80 s | 5.00 s |  | yes |

### Goal 2: K. Kraus (1. FC Kaiserslautern), shot at about 49:50.5 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | M. Wunderlich | CornerKick | 49:46.6 | – | – |  | no (lag only) |
| 2 | K. Kraus | shot | 49:50.5 | 4.60 s | 3.92 s |  | yes |

### Goal 3: P. Klement (1. FC Kaiserslautern), shot at about 95:50.1 on the clock (half 2)

| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped | Gap, your timing | Snapped? |
|---|---|---|---|---|---|---|---|
| 1 | P. Klement | Penalty | 95:50.1 | – | – |  | no (lag only) |

