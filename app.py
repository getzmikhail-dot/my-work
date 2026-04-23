import math as _math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import quote as _urlencode

import requests as _http
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@asynccontextmanager
async def lifespan(_app: 'FastAPI'):
    # При старте сервера предобучаем модели на synthetic_10000.xlsx, чтобы
    # после wake-up на бесплатном хостинге What-If/Risk/Predict сразу работали.
    _autoload_demo()
    yield


app = FastAPI(title='Employment ML Dashboard', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory='static'), name='static')

# In-memory model store (single-user local server)
_trained: Dict[str, Any] = {}
_best_model_name: Optional[str] = None
_feature_cols: List[str] = []
_unique_vals: Dict[str, Any] = {}
_stored_X: Optional[pd.DataFrame] = None      # полный набор признаков (с job) — для аналитики
_stored_X_ml: Optional[pd.DataFrame] = None   # признаки для ML (без job)
_stored_y: Optional[pd.Series] = None


class TrainPayload(BaseModel):
    rows: List[List[Any]]


_VACANCY_SCALE = 200   # число вакансий, при котором рыночный балл = 1.0
_MARKET_FACTOR_MIN = 0.7   # минимальный множитель при отсутствии вакансий
_MARKET_FACTOR_MAX = 1.15  # максимальный множитель при насыщенном рынке
_SMOOTH_PRIOR_N = 10       # «вес» априорного значения при байесовском сглаживании


def _vacancy_score(count: int) -> float:
    """Нормализует число вакансий в [0, 1] через логарифм."""
    if count <= 0:
        return 0.0
    return min(1.0, _math.log(count + 1) / _math.log(_VACANCY_SCALE + 1))


def _market_factor(count: int) -> float:
    """Мягкий множитель рынка ∈ [_MARKET_FACTOR_MIN, _MARKET_FACTOR_MAX].
    0 вакансий → 0.7 (понижаем), много вакансий → ~1.15 (немного повышаем).
    """
    s = _vacancy_score(count)
    return _MARKET_FACTOR_MIN + (_MARKET_FACTOR_MAX - _MARKET_FACTOR_MIN) * s


def _smooth(p_model: float, p_prior: float, coverage: int) -> float:
    """Байесовское сглаживание: при малом coverage тянем к p_prior.
    При coverage >> _SMOOTH_PRIOR_N → p_model; при coverage=0 → p_prior.
    """
    if coverage < 0:
        coverage = 0
    w_model = coverage / (coverage + _SMOOTH_PRIOR_N)
    return w_model * p_model + (1.0 - w_model) * p_prior


class PredictPayload(BaseModel):
    features: Dict[str, Any]
    model_name: Optional[str] = None
    ignored_cols: List[str] = []
    vacancy_count: int = -1  # -1 означает «не передано»


class RiskPayload(BaseModel):
    threshold: float = 0.4
    top_n: int = 50
    model_name: Optional[str] = None
    mode: str = 'individual'   # 'individual' | 'group'
    hide_empty: bool = True    # скрывать строки, где region+city+job все пустые


class VacancyPayload(BaseModel):
    direction: str = ''
    region: str = ''
    city: str = ''
    education_level: str = ''


# ── HH.ru integration ────────────────────────────────────────────
_HH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'ru-RU,ru;q=0.9',
}
_HH_BASE = 'https://api.hh.ru'

# Подробный словарь ключевых слов по 6-значному коду ФГОС.
# Используется и для фильтрации релевантных вакансий (_is_job_relevant), и для
# построения поискового запроса на HH.ru (_direction_to_keywords).
# Каждое ключевое слово — низкорегистровая подстрока для match по названию должности.
_DIR_KEYWORDS: Dict[str, List[str]] = {
    # 01.xx Математика и механика
    '01.03.02': ['математик', 'аналитик данных', 'data', 'разработчик', 'программист',
                 'developer', 'devops', 'python', 'java', 'frontend', 'backend',
                 'engineer', 'системный аналитик'],
    '01.04.02': ['математик', 'аналитик данных', 'data', 'разработчик', 'программист',
                 'developer', 'devops', 'научный сотрудник', 'исследователь',
                 'data scientist', 'engineer', 'архитектор'],
    # 02.xx Компьютерные и информационные науки
    '02.03.02': ['программист', 'разработчик', 'developer', 'инженер-программист',
                 'аналитик', 'аналитик 1с', '1с', 'кибербезопасность',
                 'информационная безопасность', 'devops', 'qa', 'тестировщик',
                 'техническая поддержка', 'help desk', 'data', 'python', 'java'],
    # 03.xx Физика и астрономия
    '03.03.02': ['физик', 'инженер', 'лаборант', 'научный сотрудник', 'исследователь',
                 'стажер-исследователь', 'преподаватель', 'техник', 'наладка',
                 'инженер-исследователь', 'инженер по наладке'],
    '03.04.02': ['физик', 'инженер', 'лаборант', 'научный сотрудник', 'исследователь',
                 'стажер-исследователь', 'преподаватель', 'техник', 'наладка',
                 'инженер-исследователь', 'инженер по наладке'],
    # 04.xx Химия
    '04.03.01': ['химик', 'лаборант', 'технолог', 'инженер-химик',
                 'химико-аналитический', 'хроматограф', 'отк', 'контролер отк',
                 'биохимик', 'провизор'],
    '04.03.02': ['химик', 'лаборант', 'технолог', 'инженер-химик', 'материаловед',
                 'химико-аналитический', 'хроматограф', 'материал', 'физика материалов'],
    '04.04.01': ['химик', 'лаборант', 'технолог', 'инженер-химик',
                 'химико-аналитический', 'хроматограф', 'отк', 'биохимик',
                 'научный сотрудник', 'исследователь'],
    # 05.xx Науки о Земле / Экология
    '05.03.06': ['эколог', 'экологический', 'природопользование', 'окружающая среда',
                 'охрана окружающей', 'лаборант', 'инспектор по экологии',
                 'техник-эколог', 'инженер-эколог'],
    '05.04.06': ['эколог', 'экологический', 'природопользование', 'окружающая среда',
                 'охрана окружающей', 'лаборант', 'инспектор по экологии',
                 'техник-эколог', 'инженер-эколог', 'научный сотрудник'],
    # 09.xx Информатика и вычислительная техника
    '09.03.01': ['программист', 'разработчик', 'developer', 'инженер-программист',
                 'системный администратор', 'сисадмин', 'сетевой инженер',
                 'devops', 'it-специалист', 'engineer', 'архитектор', 'qa'],
    '09.03.02': ['программист', 'разработчик', 'developer', 'аналитик',
                 'системный аналитик', 'бизнес-аналитик', '1с', 'devops', 'qa',
                 'тестировщик', 'engineer', 'data'],
    '09.03.03': ['программист', 'разработчик', 'developer', 'аналитик', '1с',
                 'аналитик 1с', '1с-разработчик', 'devops', 'qa', 'тестировщик',
                 'data', 'инженер-программист'],
    '09.03.04': ['программист', 'разработчик', 'developer', 'engineer', 'frontend',
                 'backend', 'fullstack', 'devops', 'qa', 'тестировщик', 'software',
                 'java', 'python', 'php', 'c++', 'c#', '.net'],
    '09.04.03': ['программист', 'разработчик', 'developer', 'аналитик', '1с',
                 'devops', 'qa', 'data', 'scientist', 'engineer', 'архитектор',
                 'ведущий разработчик'],
    # 11.xx Электроника, радиотехника
    '11.03.03': ['инженер', 'конструктор', 'техник', 'монтажник', 'регулировщик',
                 'электроник', 'радиотехник', 'радиоэлектроник', 'инженер рэс',
                 'сервис-инженер', 'инженер по качеству', 'инженер-конструктор'],
    '11.04.03': ['инженер', 'конструктор', 'техник', 'монтажник', 'регулировщик',
                 'электроник', 'радиотехник', 'радиоэлектроник', 'инженер рэс',
                 'сервис-инженер', 'инженер по качеству', 'инженер-конструктор',
                 'ведущий инженер'],
    # 13.xx Электро- и теплоэнергетика
    '13.03.02': ['энергетик', 'инженер-энергетик', 'электромонтер', 'электротехник',
                 'подстанция', 'электроэнергетика', 'инженер-проектировщик',
                 'инженер по эксплуатации', 'электрик', 'инженер'],
    # 14.xx Ядерная энергетика и технологии
    '14.03.02': ['ядерн', 'физик', 'физик-ядерщик', 'реактор', 'атомн',
                 'радиационн', 'дозиметрист', 'лаборант', 'инженер-исследователь',
                 'оператор ядерной', 'радиационная безопасность', 'инженер'],
    # 15.xx Машиностроение
    '15.03.04': ['автоматизация', 'асу', 'асу тп', 'кипиа', 'инженер', 'технолог',
                 'автоматчик', 'инженер-автоматчик', 'инженер-технолог',
                 'инженер по автоматизации', 'мехатроник', 'робототехник',
                 'программист асу'],
    # 21.xx Прикладная геология, горное дело
    '21.05.03': ['геолог', 'геофизик', 'геологический', 'сейсмический', 'буровой',
                 'инженер-геолог', 'техник-геолог', 'техник-технолог', 'разведка',
                 'сейсмика', 'интерпретации сейсмических'],
    # 24.xx Авиационная и ракетно-космическая техника
    '24.03.04': ['авиа', 'самолет', 'авиастроение', 'аэродром', 'инженер',
                 'конструктор', 'технолог', 'инженер-конструктор',
                 'техник-конструктор', 'инженер по качеству', 'аэродромный'],
    # 27.xx Управление в технических системах
    '27.04.03': ['аналитик', 'системный аналитик', 'бизнес-аналитик', 'программист',
                 'разработчик', 'data', 'scientist', 'engineer',
                 'инженер-программист', 'ведущий аналитик'],
    # 37.xx Психологические науки
    '37.03.01': ['психолог', 'педагог-психолог', 'психолог-консультант', 'hr',
                 'hr-специалист', 'подбор персонала', 'специалист по подбору',
                 'консультант'],
    '37.04.01': ['психолог', 'педагог-психолог', 'психолог-консультант', 'hr',
                 'hr-специалист', 'подбор персонала', 'преподаватель психологии',
                 'консультант'],
    '37.05.01': ['клинический психолог', 'психолог', 'медицинский психолог',
                 'психолог-консультант', 'педагог-психолог', 'психотерапевт'],
    # 38.xx Экономика и управление
    '38.03.01': ['экономист', 'финансист', 'бухгалтер', 'аудитор',
                 'финансовый аналитик', 'кредитный менеджер', 'младший экономист',
                 'налоговый', 'казначей', 'финансовый'],
    '38.03.04': ['менеджер', 'руководитель', 'специалист', 'администратор',
                 'государственная служба', 'муниципальный', 'помощник руководителя',
                 'госслужащий', 'управляющий', 'начальник'],
    '38.03.05': ['бизнес-аналитик', 'аналитик', 'системный аналитик',
                 'продукт-менеджер', 'product manager', 'project manager',
                 'it-аналитик', 'data', 'младший аналитик', 'специалист'],
    # 39.xx Социология и социальная работа
    '39.03.01': ['социолог', 'социология', 'аналитик', 'hr',
                 'специалист отдела кадров', 'менеджер по персоналу',
                 'специалист центра занятости'],
    '39.03.02': ['социальный работник', 'специалист по социальной работе',
                 'социальный педагог', 'педагог', 'специалист центра занятости',
                 'социальная работа', 'куратор'],
    '39.04.01': ['социолог', 'социология', 'аналитик', 'hr',
                 'специалист отдела кадров', 'менеджер по персоналу', 'руководитель'],
    # 40.xx Юриспруденция
    '40.03.01': ['юрист', 'юридический', 'юрисконсульт', 'адвокат', 'прокурор',
                 'суд', 'нотариус', 'секретарь суда', 'помощник прокурора',
                 'специалист по вопросам миграции', 'инспектор', 'эксперт',
                 'правовед'],
    '40.04.01': ['юрист', 'юридический', 'юрисконсульт', 'адвокат', 'прокурор',
                 'суд', 'нотариус', 'секретарь суда', 'помощник прокурора',
                 'специалист по вопросам миграции', 'инспектор', 'эксперт',
                 'правовед', 'судья'],
    # 44.xx Образование и педагогические науки
    '44.04.01': ['учитель', 'педагог', 'преподаватель', 'воспитатель', 'тьютор',
                 'методист', 'куратор', 'педагог дополнительного образования',
                 'педагог-психолог'],
    # 45.xx Языкознание и литературоведение
    '45.03.02': ['переводчик', 'лингвист', 'преподаватель английского',
                 'учитель английского', 'редактор', 'копирайтер', 'филолог',
                 'лингвистика', 'локализация', 'контент-менеджер'],
}


def _get_direction_keywords(direction: str) -> List[str]:
    """Возвращает список ключевых слов (lowercase) для направления.
    Точное совпадение по 6-значному коду; fallback — union по 2-значному префиксу.
    Пустой список = не знаем направление, фильтр релевантности отключён.
    """
    if not direction:
        return []
    s = str(direction).strip()
    m = re.match(r'^(\d{2}\.\d{2}\.\d{2})', s)
    if m:
        kws = _DIR_KEYWORDS.get(m.group(1))
        if kws:
            return kws
    m2 = re.match(r'^(\d{2})\.', s)
    if m2:
        prefix = m2.group(1) + '.'
        merged: List[str] = []
        seen = set()
        for code, kws in _DIR_KEYWORDS.items():
            if code.startswith(prefix):
                for kw in kws:
                    if kw not in seen:
                        seen.add(kw)
                        merged.append(kw)
        return merged
    return []


def _is_job_relevant(job_title: str, direction: str) -> bool:
    """True, если название должности содержит хотя бы одно ключевое слово направления.
    Если ключевых слов для направления нет — возвращает True (не фильтруем).
    """
    if not job_title or not direction:
        return False
    keywords = _get_direction_keywords(direction)
    if not keywords:
        return True
    t = str(job_title).strip().lower().replace('ё', 'е')
    return any(kw in t for kw in keywords)

# Категории должностей — ключевые слова для классификации
_JOB_CAT_RULES: List[tuple] = [
    ('ИТ', [
        'программист', 'разработчик', 'developer', '1с', 'тестировщик', 'devops',
        'аналитик данных', 'data', 'системный администратор', 'сисадмин',
        'it-специалист', 'инженер-программист', 'веб-', 'frontend', 'backend',
        'fullstack', 'техническая поддержка', 'help desk', 'sql', 'java', 'python',
        'c++', 'c#', '.net', 'php', 'сетевой инженер', 'информационная безопасность',
    ]),
    ('Экономика и финансы', [
        'экономист', 'бухгалтер', 'финансист', 'аудитор', 'финансовый',
        'бухгалтерия', 'казначей', 'актуарий', 'налоговый', 'кредитный',
    ]),
    ('Управление', [
        'менеджер', 'руководитель', 'директор', 'управляющий', 'начальник',
        'hr', 'управление персоналом', 'project manager', 'product manager', 'супервайзер',
    ]),
    ('Маркетинг и реклама', [
        'маркетолог', 'pr-специалист', 'smm', 'рекламист', 'контент-менеджер',
        'бренд', 'копирайтер', 'seo',
    ]),
    ('Образование', [
        'учитель', 'преподаватель', 'педагог', 'воспитатель', 'тренер',
        'методист', 'репетитор', 'куратор',
    ]),
    ('Юриспруденция', [
        'юрист', 'адвокат', 'правовед', 'нотариус', 'следователь',
        'прокурор', 'судья', 'юрисконсульт',
    ]),
    ('Медицина', [
        'врач', 'медсестра', 'фармацевт', 'стоматолог', 'фельдшер',
        'медицинский', 'медбрат', 'санитар', 'лаборант', 'провизор',
    ]),
    ('Строительство', [
        'строитель', 'архитектор', 'прораб', 'проектировщик',
        'инженер-строитель', 'сметчик', 'монтажник',
    ]),
    ('Торговля и продажи', [
        'продавец', 'торговый представитель', 'мерчандайзер',
        'байер', 'закупщик', 'менеджер по продажам',
    ]),
    ('Логистика', [
        'логист', 'снабженец', 'складской', 'транспортный',
        'диспетчер', 'экспедитор',
    ]),
    ('Инженерия', [
        'инженер', 'конструктор', 'технолог', 'механик', 'электрик',
        'электромонтажник', 'слесарь', 'сварщик',
    ]),
]


def _classify_job(title: str) -> str:
    if not title:
        return 'Другое'
    t = str(title).strip().lower().replace('ё', 'е')
    for cat, keywords in _JOB_CAT_RULES:
        if any(kw in t for kw in keywords):
            return cat
    return 'Другое'


# Категории направлений по двузначному префиксу кода ФГОС
_DIRECTION_CATEGORIES: Dict[str, str] = {
    '01': 'ИТ', '02': 'ИТ', '09': 'ИТ', '10': 'ИТ', '27': 'ИТ',
    '11': 'Инженерия', '12': 'Инженерия', '13': 'Инженерия', '14': 'Инженерия',
    '15': 'Инженерия', '16': 'Инженерия', '17': 'Инженерия', '18': 'Инженерия',
    '19': 'Инженерия', '20': 'Инженерия', '21': 'Инженерия', '22': 'Инженерия',
    '23': 'Инженерия', '24': 'Инженерия', '25': 'Инженерия', '26': 'Инженерия',
    '28': 'Инженерия', '29': 'Инженерия',
    '30': 'Медицина', '31': 'Медицина', '32': 'Медицина', '33': 'Медицина', '34': 'Медицина',
    '35': 'Аграрные науки', '36': 'Аграрные науки',
    '37': 'Психология',
    '38': 'Экономика', '39': 'Социология',
    '40': 'Юриспруденция', '41': 'Политология',
    '42': 'Журналистика',
    '43': 'Туризм',
    '44': 'Образование',
    '45': 'Лингвистика', '46': 'История', '47': 'Философия',
    '49': 'Спорт',
    '51': 'Культура', '52': 'Культура', '53': 'Музыка', '54': 'Дизайн',
}


def _get_direction_category(direction: str) -> str:
    m = re.match(r'^(\d{2})\.', str(direction).strip())
    if m:
        return _DIRECTION_CATEGORIES.get(m.group(1), 'Другое')
    return 'Другое'


# Маппинг ключевых слов названий регионов -> двузначный код ОКТМО
_REGION_OKTMO: Dict[str, str] = {
    'адыгея': '01', 'башкортостан': '02', 'башкирия': '02',
    'бурятия': '03', 'алтай республика': '04', 'дагестан': '05',
    'ингушетия': '06', 'кабардино': '07', 'калмыкия': '08',
    'карачаево': '09', 'карелия': '10', 'коми': '11',
    'марий эл': '12', 'мордовия': '13', 'саха': '14', 'якутия': '14',
    'северная осетия': '15', 'татарстан': '16', 'татария': '16',
    'тыва': '17', 'тува': '17', 'удмуртия': '18', 'хакасия': '19',
    'чечня': '20', 'чеченская': '20', 'чувашия': '21',
    'алтайский': '22', 'краснодарский': '23', 'кубань': '23',
    'красноярский': '24', 'приморский': '25', 'ставропольский': '26',
    'хабаровский': '27', 'амурская': '28', 'архангельская': '29',
    'астраханская': '30', 'белгородская': '31', 'брянская': '32',
    'владимирская': '33', 'волгоградская': '34', 'вологодская': '35',
    'воронежская': '36', 'ивановская': '37', 'иркутская': '38',
    'калининградская': '39', 'калужская': '40', 'камчатский': '41',
    'кемеровская': '42', 'кировская': '43', 'костромская': '44',
    'курганская': '45', 'курская': '46', 'ленинградская': '47',
    'липецкая': '48', 'магаданская': '49',
    'московская': '50', 'подмосковье': '50',
    'мурманская': '51', 'нижегородская': '52', 'новгородская': '53',
    'новосибирская': '54', 'омская': '55', 'оренбургская': '56',
    'орловская': '57', 'пензенская': '58', 'пермский': '59',
    'псковская': '60', 'ростовская': '61', 'рязанская': '62',
    'самарская': '63', 'саратовская': '64', 'сахалинская': '65',
    'свердловская': '66', 'смоленская': '67', 'тамбовская': '68',
    'тверская': '69', 'томская': '70', 'тульская': '71',
    'тюменская': '72', 'ульяновская': '73', 'челябинская': '74',
    'забайкальский': '75', 'ярославская': '76',
    'москва': '77', 'санкт-петербург': '78', 'петербург': '78', 'спб': '78',
    'еврейская': '79', 'ненецкий': '83',
    'ханты-мансийский': '86', 'хмао': '86', 'югра': '86',
    'чукотский': '87', 'ямало-ненецкий': '89', 'янао': '89',
}
# Аббревиатуры - точное совпадение
_REGION_ABBREV: Dict[str, str] = {
    'мо': '50', 'ло': '47', 'ко': '44', 'то': '69',
    'но': '53', 'во': '36', 'ио': '38', 'ро': '61',
}
_TRUDVSEM_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
_TRUDVSEM_BASE = 'https://opendata.trudvsem.ru/api/v1/vacancies'


def _find_trudvsem_region(name: str) -> Optional[str]:
    if not name:
        return None
    nl = name.strip().lower().replace('ё', 'е')
    code = _REGION_ABBREV.get(nl)
    if not code:
        for kw, c in _REGION_OKTMO.items():
            if kw in nl:
                code = c
                break
    return (code + '00000000000') if code else None


def _count_trudvsem_vacancies(job_title: str, region_oktmo: Optional[str]) -> int:
    try:
        if region_oktmo:
            url = f'{_TRUDVSEM_BASE}/region/{region_oktmo}'
        else:
            url = _TRUDVSEM_BASE
        resp = _http.get(url, params={'text': job_title, 'limit': 1},
                         headers=_TRUDVSEM_HEADERS, timeout=10)
        if resp.status_code == 200:
            return int(resp.json().get('meta', {}).get('total', 0))
    except Exception:
        return -1
    return -1


_hh_areas_cache: Dict[str, str] = {}   # {нормализованное_имя: area_id}
_hh_areas_ts: float = 0.0
_HH_AREAS_TTL = 86400.0  # обновляем раз в сутки


def _nk_hh(s: str) -> str:
    return s.strip().lower().replace('ё', 'е')


def _load_hh_areas() -> Dict[str, str]:
    global _hh_areas_cache, _hh_areas_ts
    if _hh_areas_cache and (time.time() - _hh_areas_ts) < _HH_AREAS_TTL:
        return _hh_areas_cache
    try:
        resp = _http.get(f'{_HH_BASE}/areas', headers=_HH_HEADERS, timeout=8)
        resp.raise_for_status()

        flat: Dict[str, str] = {}

        def traverse(areas: List[Dict]) -> None:
            for a in areas:
                flat[_nk_hh(a['name'])] = str(a['id'])
                if a.get('areas'):
                    traverse(a['areas'])

        traverse(resp.json())
        _hh_areas_cache = flat
        _hh_areas_ts = time.time()
    except Exception as _e:
        print(f'[hh-areas] {_e}')
    return _hh_areas_cache


def _find_area_id(name: str) -> Optional[str]:
    areas = _load_hh_areas()
    key = _nk_hh(name)
    if not key:
        return None
    if key in areas:
        return areas[key]
    # подстрочное совпадение — берём самый длинный совпавший ключ
    best_id, best_len = None, 0
    for k, v in areas.items():
        if (k in key or key in k) and len(k) > best_len:
            best_len, best_id = len(k), v
    return best_id


def _direction_to_keywords(direction: str) -> str:
    """Строит короткую строку для поискового запроса HH.ru — топ-4 ключевых слова."""
    kws = _get_direction_keywords(direction)
    if kws:
        return ' '.join(kws[:4])
    return re.sub(r'^\d+\.\d+\.\d+\s*', '', direction).strip()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        low = str(c).strip().lower().replace('ё', 'е')
        if 'уров' in low and 'образ' in low:
            mapping[c] = 'education_level'
        elif 'направ' in low:
            mapping[c] = 'direction'
        elif 'регион' in low:
            mapping[c] = 'region'
        elif 'город' in low:
            mapping[c] = 'city'
        elif 'должност' in low or 'кем вы работаете' in low:
            mapping[c] = 'job'
        elif 'работаете по специальности' in low or 'специальности' in low or 'employed' in low or 'target' in low:
            mapping[c] = 'target'
    return df.rename(columns=mapping)


def normalize_target(v: Any):
    # Числовые значения (в т.ч. '2.0', '1.0', '0.0' после повторного препроцессинга)
    try:
        n = float(str(v).strip())
        if abs(n - 2) < 0.01:
            return 2
        if abs(n - 1) < 0.01:
            return 1
        return 0
    except (ValueError, TypeError):
        pass
    s = str(v).strip().lower().replace('ё', 'е')
    if s in {'да', 'yes', 'true', 'д'} or s.startswith('да'):
        return 2  # трудоустроен по специальности
    if s in {'нет', 'no', 'false', 'н'} or s.startswith('нет'):
        return 1  # трудоустроен не по специальности
    return 0  # не трудоустроен / NaN / неизвестно


def clean_empty(v: Any):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == '' or s.lower() in {'nan', 'none', 'null', 'na', 'n/a', '-'}:
        return np.nan
    return v


def detect_and_cast_numeric(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        converted = pd.to_numeric(
            result[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        original_non_null = result[col].notna().sum()
        ratio = converted.notna().sum() / original_non_null if original_non_null else 0.0
        if ratio >= threshold:
            result[col] = converted
    return result


def preprocess_dataframe(rows: List[List[Any]]):
    empty_meta = {
        'rows_after_cleaning': 0,
        'original_fill_percent': 0.0,
        'processed_fill_percent': 0.0,
        'filled_missing_count': 0,
        'target_column': None,
        'numeric_columns': [],
        'categorical_columns': [],
    }

    if not rows or len(rows) < 2:
        return rows, empty_meta

    headers = [str(x).strip() for x in rows[0]]
    data = pd.DataFrame(rows[1:], columns=headers)
    total_cells = max(len(data.index) * len(data.columns), 1)

    original_filled_mask = data.notna() & (data.astype(str).apply(lambda col: col.str.strip()) != '')
    original_fill = float(original_filled_mask.sum().sum() / total_cells * 100)

    data = data.apply(lambda col: col.map(clean_empty))
    data = normalize_cols(data)

    target_col = 'target' if 'target' in data.columns else None
    if target_col:
        data[target_col] = data[target_col].map(normalize_target)

    data = detect_and_cast_numeric(data)

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    filled_missing = 0

    for col in numeric_cols:
        miss = int(data[col].isna().sum())
        if miss:
            median = data[col].median()
            if pd.isna(median):
                median = 0
            data[col] = data[col].fillna(median)
            filled_missing += miss

    for col in categorical_cols:
        miss = int(data[col].isna().sum())
        if miss:
            data[col] = data[col].fillna('Не указано')
            filled_missing += miss

    processed_fill = float(data.notna().sum().sum() / total_cells * 100)
    out_rows = [data.columns.tolist()] + data.astype(object).where(pd.notna(data), '').values.tolist()

    meta = {
        'rows_after_cleaning': int(len(data)),
        'original_fill_percent': round(original_fill, 1),
        'processed_fill_percent': round(processed_fill, 1),
        'filled_missing_count': int(filled_missing),
        'target_column': target_col,
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
    }
    return out_rows, meta


def _autoload_demo() -> None:
    """Вызывается из lifespan-хука при старте сервера."""
    import os
    candidates = ['synthetic_10000.xlsx', 'static/synthetic_10000.xlsx']
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print('[startup] synthetic_10000.xlsx не найден — пропускаю автотренировку')
        return
    try:
        df = pd.read_excel(path)
        rows = [df.columns.tolist()] + df.values.tolist()
        result = train(TrainPayload(rows=rows))
        print(f"[startup] автотренировка на {len(df)} строках OK, best={result.get('best_model')}")
    except Exception as e:
        print(f'[startup] автотренировка не удалась: {e}')


@app.get('/')
def root():
    return FileResponse('static/dashboard.html')


@app.post('/api/preprocess')
def preprocess(payload: TrainPayload):
    processed_rows, meta = preprocess_dataframe(payload.rows)
    return {'rows': processed_rows or payload.rows, 'meta': meta}


@app.post('/api/train')
def train(payload: TrainPayload):
    global _trained, _best_model_name, _feature_cols, _unique_vals, _stored_X, _stored_X_ml, _stored_y

    rows = payload.rows
    if not rows or len(rows) < 3:
        return {'models': [], 'best_model': None, 'feature_importance': [], 'meta': {'error': 'not_enough_rows'}}

    processed_rows, prep_meta = preprocess_dataframe(rows)
    if not processed_rows or len(processed_rows) < 2:
        return {'models': [], 'best_model': None, 'feature_importance': [], 'meta': {'error': 'empty_after_preprocessing'}}

    headers = [str(x).strip() for x in processed_rows[0]]
    data = pd.DataFrame(processed_rows[1:], columns=headers)
    data = data.replace('', np.nan)

    if 'target' not in data.columns:
        return {'models': [], 'best_model': None, 'feature_importance': [], 'meta': {'error': 'target_not_found', 'preprocess': prep_meta}}

    data['target'] = pd.to_numeric(data['target'], errors='coerce')
    data = data.dropna(subset=['target']).copy()
    if len(data) < 20:
        return {'models': [], 'best_model': None, 'feature_importance': [], 'meta': {'error': 'less_than_20_rows', 'preprocess': prep_meta}}

    y = data['target'].astype(int)
    X = data.drop(columns=['target']).copy()
    X = detect_and_cast_numeric(X)

    if 'education_level' in X.columns:
        def _norm_edu(v):
            s = str(v).strip().lower().replace('ё', 'е')
            if 'бакалав' in s:
                return 'Бакалавриат'
            if 'магистр' in s:
                return 'Магистратура'
            if 'специал' in s:
                return 'Специалитет'
            return str(v).strip()
        X['education_level'] = X['education_level'].apply(_norm_edu)

    # job — результат трудоустройства, не предсказатель; убираем из ML, оставляем для аналитики
    X_ml = X.drop(columns=['job'], errors='ignore')

    num_cols = X_ml.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_ml.columns if c not in num_cols]
    if not num_cols and not cat_cols:
        return {'models': [], 'best_model': None, 'feature_importance': [], 'meta': {'error': 'no_features', 'preprocess': prep_meta}}

    transformers = []
    if cat_cols:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]),
            cat_cols,
        ))
    if num_cols:
        transformers.append((
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            num_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return {'models': [], 'best_model': None, 'feature_importance': [], 'meta': {'error': 'single_class_target', 'preprocess': prep_meta}}

    cv_folds = max(2, min(5, int(class_counts.min())))

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'LightGBM': lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=300, random_state=42, verbose=0),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    best_name, best_cv, best_pipe = None, -1.0, None
    all_pipes: Dict[str, Any] = {}

    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        cv = float(cross_val_score(pipe, X_ml, y, cv=cv_folds, scoring='f1_macro').mean()) if cv_folds >= 2 else 0.0

        cal_pipe = CalibratedClassifierCV(pipe, method='isotonic', cv=min(3, cv_folds))
        cal_pipe.fit(X_train, y_train)
        pred = cal_pipe.predict(X_test)
        proba_all = cal_pipe.predict_proba(X_test)

        all_pipes[name] = cal_pipe

        n_classes = len(set(y_test))
        item = {
            'name': name,
            'accuracy': float(accuracy_score(y_test, pred)),
            'f1': float(f1_score(y_test, pred, average='macro', zero_division=0)),
            'precision': float(precision_score(y_test, pred, average='macro', zero_division=0)),
            'recall': float(recall_score(y_test, pred, average='macro', zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, proba_all, multi_class='ovr', average='macro')) if n_classes > 1 else 0.0,
            'cv_f1': cv,
        }
        results.append(item)

        if cv > best_cv:
            best_cv, best_name, best_pipe = cv, name, pipe

    if best_pipe is None and results:
        best_name = max(results, key=lambda x: x['f1'])['name']
        best_pipe = all_pipes[best_name]

    # Человекочитаемые названия колонок
    _DISPLAY = {
        'education_level': 'Уровень образования',
        'direction': 'Направление',
        'region': 'Регион',
        'city': 'Город',
        'job': 'Должность',
    }

    def pipe_importance(pipe: Any) -> List[Dict]:
        try:
            actual = pipe.calibrated_classifiers_[0].estimator if hasattr(pipe, 'calibrated_classifiers_') else pipe
            prep_step = actual.named_steps['prep']
            feat_names = prep_step.get_feature_names_out()
            model_obj  = actual.named_steps['model']
            raw = None
            if hasattr(model_obj, 'feature_importances_'):
                raw = model_obj.feature_importances_
            elif hasattr(model_obj, 'coef_'):
                raw = np.abs(model_obj.coef_[0])
            if raw is None:
                return []
            col_imp: Dict[str, float] = {}
            for fname, fval in zip(feat_names, raw):
                for col in cat_cols:
                    if fname.startswith('cat__' + col + '_'):
                        col_imp[col] = col_imp.get(col, 0.0) + float(fval)
                        break
                else:
                    for col in num_cols:
                        if fname == 'num__' + col:
                            col_imp[col] = col_imp.get(col, 0.0) + float(fval)
                            break
            total = sum(col_imp.values()) or 1.0
            return sorted(
                [{'label': _DISPLAY.get(k, k), 'pct': round(v / total * 100, 1)}
                 for k, v in col_imp.items()],
                key=lambda x: x['pct'], reverse=True
            )
        except Exception:
            return []

    all_feature_importances = {name: pipe_importance(pipe) for name, pipe in all_pipes.items()}
    feature_importance = all_feature_importances.get(best_name or '', [])

    # Store trained models and metadata globally
    _trained = all_pipes
    _best_model_name = best_name
    _feature_cols = num_cols + cat_cols
    _stored_X = X.copy()        # с job — для аналитики вакансий
    _stored_X_ml = X_ml.copy()  # без job — для ML предсказаний
    _stored_y = y.copy()

    # Unique values for What-If form
    unique_vals: Dict[str, Any] = {}
    for col in cat_cols:
        vals = sorted(X_ml[col].dropna().astype(str).unique().tolist())
        unique_vals[col] = vals
    for col in num_cols:
        unique_vals[col] = {
            'min': round(float(X_ml[col].min()), 2),
            'max': round(float(X_ml[col].max()), 2),
            'median': round(float(X_ml[col].median()), 2),
        }
    _unique_vals = unique_vals

    # direction ↔ education_level mapping
    dir_by_level: Dict[str, List[str]] = {}
    if 'education_level' in X.columns and 'direction' in X.columns:
        for level in X['education_level'].dropna().unique():
            mask = X['education_level'] == level
            dirs = sorted(X['direction'][mask].dropna().astype(str).unique().tolist())
            dir_by_level[str(level)] = dirs

    # Топ-должности успешно трудоустроенных по каждому направлению
    direction_top_jobs: Dict[str, List[str]] = {}
    if 'direction' in X.columns and 'job' in X.columns:
        for direction in X['direction'].dropna().unique():
            mask = (X['direction'].astype(str) == str(direction)) & (y == 1)
            jobs = X.loc[mask, 'job'].dropna().astype(str)
            if len(jobs):
                direction_top_jobs[str(direction)] = jobs.value_counts().head(5).index.tolist()

    # Категория для каждой уникальной должности
    job_categories: Dict[str, str] = {}
    if 'job' in X.columns:
        for job in X['job'].dropna().astype(str).unique():
            job_categories[job] = _classify_job(job)

    # Города по регионам из загруженных данных
    region_cities: Dict[str, List[str]] = {}
    _empty = {'не указано', 'nan', 'none', '', 'нет', 'н/д'}
    if 'region' in X.columns and 'city' in X.columns:
        for region in X['region'].dropna().astype(str).unique():
            if region.lower().replace('ё', 'е') in _empty:
                continue
            mask = X['region'].astype(str) == region
            cities = sorted(
                c for c in X['city'][mask].dropna().astype(str).unique()
                if c.lower().replace('ё', 'е') not in _empty
            )
            if cities:
                region_cities[region] = cities

    results.sort(key=lambda x: x['cv_f1'], reverse=True)
    return {
        'models': results,
        'best_model': best_name,
        'feature_importance': feature_importance,
        'all_feature_importances': all_feature_importances,
        'feature_cols': _feature_cols,
        'unique_vals': unique_vals,
        'dir_by_level': dir_by_level,
        'direction_top_jobs': direction_top_jobs,
        'job_categories': job_categories,
        'region_cities': region_cities,
        'meta': {
            'rowsAfterCleaning': int(len(data)),
            'numericColumns': num_cols,
            'categoricalColumns': cat_cols,
            'cvFolds': int(cv_folds),
            'preprocess': prep_meta,
        },
    }


@app.post('/api/predict')
def predict(payload: PredictPayload):
    name = payload.model_name or _best_model_name
    if not name or name not in _trained:
        return {'error': 'model_not_trained'}
    pipe = _trained[name]
    try:
        ignored = set(payload.ignored_cols)
        features = dict(payload.features)
        for col in _feature_cols:
            if col not in features or col in ignored:
                uv = _unique_vals.get(col, {})
                features[col] = uv.get('median', 0.0) if isinstance(uv, dict) else 'Не указано'
        row = pd.DataFrame([features])
        row = detect_and_cast_numeric(row)
        proba_all_row = pipe.predict_proba(row)[0]
        classes = list(pipe.classes_)
        p_spec_raw = float(proba_all_row[classes.index(2)]) if 2 in classes else 0.0
        p_emp_raw = float(sum(proba_all_row[classes.index(c)] for c in [1, 2] if c in classes))

        # Coverage и подсчёт похожих записей в выборке
        coverage: Optional[int] = None
        cov_employed: int = 0
        cov_specialty: int = 0
        if _stored_X is not None and _stored_y is not None:
            active_cols = [
                col for col in _feature_cols
                if col not in ignored
                and col in _stored_X.columns
                and col in payload.features
                and _stored_X[col].dtype == object
            ]
            if active_cols:
                mask = pd.Series(True, index=_stored_X.index)
                for col in active_cols:
                    mask &= _stored_X[col].astype(str) == str(payload.features[col])
                coverage = int(mask.sum())
                cov_employed = int((_stored_y[mask] > 0).sum())
                cov_specialty = int((_stored_y[mask] == 2).sum())
            else:
                coverage = len(_stored_X)
                cov_employed = int((_stored_y > 0).sum())
                cov_specialty = int((_stored_y == 2).sum())

        # Глобальные базовые вероятности (prior) — по всей выборке
        prior_emp = 0.5
        prior_spec_given_emp = 0.5
        if _stored_y is not None and len(_stored_y) > 0:
            n_total = int(len(_stored_y))
            n_emp = int((_stored_y > 0).sum())
            n_spec = int((_stored_y == 2).sum())
            prior_emp = n_emp / n_total if n_total else 0.5
            prior_spec_given_emp = (n_spec / n_emp) if n_emp else 0.5

        # Байесовское сглаживание модели к prior'ам через coverage
        cov = coverage if coverage is not None else 0
        p_employed = _smooth(p_emp_raw, prior_emp, cov)
        # Условная вероятность «по специальности | трудоустроен»
        p_spec_given_emp_raw = (p_spec_raw / p_emp_raw) if p_emp_raw > 1e-9 else 0.0
        p_spec_given_emp = _smooth(p_spec_given_emp_raw, prior_spec_given_emp, cov_employed)
        p_specialty = p_employed * p_spec_given_emp

        # Рыночный множитель — мягкая поправка [0.7, 1.15]
        market_factor: Optional[float] = None
        v_score: Optional[float] = None
        p_employed_final = p_employed
        p_specialty_final = p_specialty
        if payload.vacancy_count >= 0:
            v_score = _vacancy_score(payload.vacancy_count)
            market_factor = _market_factor(payload.vacancy_count)
            p_employed_final = min(1.0, p_employed * market_factor)
            p_specialty_final = min(1.0, p_specialty * market_factor)

        return {
            'probability': round(p_specialty_final, 4),
            'probability_employed': round(p_employed_final, 4),
            'probability_spec_given_employed': round(p_spec_given_emp, 4),
            'probability_raw_employed': round(p_emp_raw, 4),
            'probability_raw_specialty': round(p_spec_raw, 4),
            'prior_employed': round(prior_emp, 4),
            'prior_spec_given_employed': round(prior_spec_given_emp, 4),
            'market_factor': round(market_factor, 4) if market_factor is not None else None,
            'vacancy_score': round(v_score, 4) if v_score is not None else None,
            'vacancy_count': payload.vacancy_count if payload.vacancy_count >= 0 else None,
            'model': name,
            'coverage': coverage,
            'coverage_employed': cov_employed,
            'coverage_specialty': cov_specialty,
        }
    except Exception as e:
        return {'error': str(e)}


_EMPTY_PROFILE_VALUES = {'не указано', 'nan', 'none', 'null', '', 'н/д', 'na', 'n/a', '-', 'нет'}


def _is_empty_value(v: Any) -> bool:
    return str(v).strip().lower().replace('ё', 'е') in _EMPTY_PROFILE_VALUES


def _agg_jobs(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    cleaned = sorted({v.strip() for v in vals if not _is_empty_value(v)})
    return ', '.join(cleaned) if cleaned else '—'


@app.post('/api/risk_groups')
def risk_groups(payload: RiskPayload):
    name = payload.model_name or _best_model_name
    if not name or name not in _trained or _stored_X_ml is None or _stored_X is None or _stored_y is None:
        return {'error': 'model_not_trained'}
    pipe = _trained[name]
    try:
        classes = list(pipe.classes_)
        idx2 = classes.index(2) if 2 in classes else len(classes) - 1
        probas_raw = pipe.predict_proba(_stored_X_ml)[:, idx2]

        # Глобальный prior P(спец) — для байесовского сглаживания на малых группах
        n_total = int(len(_stored_y))
        n_spec = int((_stored_y == 2).sum())
        prior_spec = n_spec / n_total if n_total else 0.5

        df = _stored_X.copy()
        df['_proba_raw'] = probas_raw
        df['_y'] = _stored_y.values

        # Маска «пустой анкеты»: все три поля (region, city, job) пустые/«Не указано»
        candidate_cols = [c for c in ('region', 'city', 'job') if c in df.columns]
        if candidate_cols:
            empty_mask = pd.Series(True, index=df.index)
            for col in candidate_cols:
                empty_mask &= df[col].astype(str).map(_is_empty_value)
        else:
            empty_mask = pd.Series(False, index=df.index)
        empty_count = int(empty_mask.sum())

        if payload.hide_empty:
            df = df.loc[~empty_mask].copy()

        if df.empty:
            return {
                'mode': payload.mode,
                'risk_count': 0,
                'total': 0,
                'total_students': 0,
                'empty_filtered': empty_count,
                'threshold': payload.threshold,
                'records': [],
            }

        if payload.mode == 'group':
            # Агрегат по «направление + уровень»
            group_cols = [c for c in ('direction', 'education_level') if c in df.columns]
            if not group_cols:
                return {'error': 'no_group_columns'}

            grouped = df.groupby(group_cols, dropna=False)
            agg = pd.DataFrame({
                'count': grouped.size(),
                'mean_proba': grouped['_proba_raw'].mean(),
                'spec_count': grouped['_y'].apply(lambda s: int((s == 2).sum())),
                'emp_count': grouped['_y'].apply(lambda s: int((s > 0).sum())),
            }).reset_index()

            agg['probability'] = agg.apply(
                lambda r: _smooth(float(r['mean_proba']), prior_spec, int(r['count'])),
                axis=1
            )

            risky = agg[agg['probability'] < payload.threshold].copy()
            risky = risky.sort_values('probability').head(payload.top_n)

            records: List[Dict[str, Any]] = []
            for _, r in risky.iterrows():
                cnt = int(r['count'])
                records.append({
                    'education_level': str(r.get('education_level', '')),
                    'direction': str(r.get('direction', '')),
                    'count': cnt,
                    'spec_count': int(r['spec_count']),
                    'emp_count': int(r['emp_count']),
                    'probability': round(float(r['probability']), 3),
                    'fact_spec_pct': round(int(r['spec_count']) / cnt * 100, 1) if cnt else 0.0,
                    'fact_emp_pct': round(int(r['emp_count']) / cnt * 100, 1) if cnt else 0.0,
                })

            return {
                'mode': 'group',
                'risk_count': int((agg['probability'] < payload.threshold).sum()),
                'total': int(len(agg)),
                'total_students': int(len(df)),
                'empty_filtered': empty_count,
                'threshold': payload.threshold,
                'records': records,
            }

        # mode == 'individual' — дедупликация по профилю (без должности)
        dedup_cols = [c for c in ('education_level', 'direction', 'region', 'city') if c in df.columns]
        if not dedup_cols:
            return {'error': 'no_dedup_columns'}

        grouped = df.groupby(dedup_cols, dropna=False)
        agg_individual = pd.DataFrame({
            'count': grouped.size(),
            'mean_proba': grouped['_proba_raw'].mean(),
            'spec_count': grouped['_y'].apply(lambda s: int((s == 2).sum())),
        })
        if 'job' in df.columns:
            agg_individual['jobs'] = grouped['job'].apply(_agg_jobs)
        else:
            agg_individual['jobs'] = '—'
        agg_individual = agg_individual.reset_index()

        agg_individual['probability'] = agg_individual.apply(
            lambda r: _smooth(float(r['mean_proba']), prior_spec, int(r['count'])),
            axis=1
        )

        risky_ind = agg_individual[agg_individual['probability'] < payload.threshold].copy()
        risky_ind = risky_ind.sort_values('probability').head(payload.top_n)

        ind_records: List[Dict[str, Any]] = []
        for _, r in risky_ind.iterrows():
            ind_records.append({
                'education_level': str(r.get('education_level', '')),
                'direction': str(r.get('direction', '')),
                'region': str(r.get('region', '')),
                'city': str(r.get('city', '')),
                'jobs': str(r.get('jobs', '—')),
                'count': int(r['count']),
                'spec_count': int(r['spec_count']),
                'probability': round(float(r['probability']), 3),
            })

        return {
            'mode': 'individual',
            'risk_count': int((agg_individual['probability'] < payload.threshold).sum()),
            'total': int(len(agg_individual)),
            'total_students': int(len(df)),
            'empty_filtered': empty_count,
            'threshold': payload.threshold,
            'records': ind_records,
        }
    except Exception as e:
        return {'error': str(e)}


@app.post('/api/vacancies')
def get_vacancies(payload: VacancyPayload):
    try:
        # Город точнее региона
        area_name = ''
        if payload.city and payload.city not in ('Не указано', ''):
            area_name = payload.city
        elif payload.region and payload.region not in ('Не указано', ''):
            area_name = payload.region

        area_id = _find_area_id(area_name) if area_name else None

        # Код региона для Trudvsem: сначала по региону, потом по городу из данных
        region_oktmo: Optional[str] = _find_trudvsem_region(payload.region)
        if not region_oktmo and _stored_X is not None and payload.city and 'city' in _stored_X.columns and 'region' in _stored_X.columns:
            city_rows = _stored_X[_stored_X['city'].astype(str).str.lower() == payload.city.lower()]
            if len(city_rows) > 0:
                region_from_data = str(city_rows['region'].mode().iloc[0])
                region_oktmo = _find_trudvsem_region(region_from_data)
        if not region_oktmo:
            region_oktmo = _find_trudvsem_region(payload.city)

        local_count = 0
        job_links: List[Dict] = []

        if _stored_X is not None and _stored_y is not None and 'direction' in _stored_X.columns:
            X = _stored_X  # локальная ссылка — type checker увидит non-None в замыканиях
            # Базовые маски (без гео и edu), чтобы можно было пошагово ослаблять фильтры
            exact_base = X['direction'].astype(str) == payload.direction
            dir_category = _get_direction_category(payload.direction)
            if dir_category != 'Другое':
                cat_base = X['direction'].astype(str).apply(_get_direction_category) == dir_category
            else:
                cat_base = exact_base.copy()

            # Опциональный фильтр по уровню образования
            edu_mask: Optional[pd.Series] = None
            if payload.education_level and 'education_level' in X.columns:
                edu_mask = X['education_level'].astype(str) == payload.education_level

            # Опциональный гео-фильтр (точное совпадение, fallback на contains)
            geo_mask: Optional[pd.Series] = None
            if area_name:
                al = area_name.strip().lower().replace('ё', 'е')
                gm = pd.Series(False, index=X.index)
                if 'region' in X.columns:
                    reg_norm = X['region'].astype(str).str.lower().str.replace('ё', 'е', regex=False)
                    exact_reg = reg_norm == al
                    gm |= exact_reg if exact_reg.any() else reg_norm.str.contains(al, regex=False, na=False)
                if 'city' in X.columns:
                    city_norm = X['city'].astype(str).str.lower().str.replace('ё', 'е', regex=False)
                    exact_city = city_norm == al
                    gm |= exact_city if exact_city.any() else city_norm.str.contains(al, regex=False, na=False)
                geo_mask = gm

            # local_count = точное направление + гео + edu, любое трудоустройство
            local_mask = exact_base.copy()
            if geo_mask is not None:
                local_mask &= geo_mask
            if edu_mask is not None:
                local_mask &= edu_mask
            local_count = int((_stored_y[local_mask] > 0).sum())

            # Подбор должностей — пошаговый fallback с фильтром релевантности _is_job_relevant.
            # Стадии (от строгой к широкой), на каждой добираем до 8 релевантных:
            #   exact + spec(y==2) + geo + edu  →  ... + edu  →  ... (без всего)
            #   cat   + emp(y>0)   + geo + edu  →  ... + edu  →  ... (без всего)
            if 'job' in X.columns:
                _EMPTY_JOB = {'не указано', 'nan', 'none', 'null', '', 'н/д', 'na', 'n/a', '-', 'нет'}

                def _clean_jobs(series: pd.Series) -> pd.Series:
                    s = series.dropna().astype(str)
                    return s[~s.str.strip().str.lower().str.replace('ё', 'е', regex=False).isin(_EMPTY_JOB)]

                MAX_TITLES = 8
                job_titles: List[str] = []
                seen: set = set()

                def _gather(base: pd.Series, target_mask: pd.Series, with_geo: bool, with_edu: bool) -> None:
                    if len(job_titles) >= MAX_TITLES:
                        return
                    m = base & target_mask
                    if with_geo and geo_mask is not None:
                        m &= geo_mask
                    if with_edu and edu_mask is not None:
                        m &= edu_mask
                    jobs = _clean_jobs(X.loc[m, 'job'])
                    for raw in jobs.value_counts().index.tolist():
                        if len(job_titles) >= MAX_TITLES:
                            return
                        title = str(raw)
                        key = title.lower()
                        if key in seen:
                            continue
                        if _is_job_relevant(title, payload.direction):
                            job_titles.append(title)
                            seen.add(key)

                spec = _stored_y == 2
                emp = _stored_y > 0

                # Точное направление, по специальности
                _gather(exact_base, spec, True, True)
                if geo_mask is not None:
                    _gather(exact_base, spec, False, True)
                if edu_mask is not None:
                    _gather(exact_base, spec, False, False)

                # ФГОС-категория, любое трудоустройство
                _gather(cat_base, emp, True, True)
                if geo_mask is not None:
                    _gather(cat_base, emp, False, True)
                if edu_mask is not None:
                    _gather(cat_base, emp, False, False)

                # Параллельный запрос счётчиков через Trudvsem (публичный API)
                counts: Dict[str, int] = {}
                with ThreadPoolExecutor(max_workers=5) as pool:
                    future_to_title = {
                        pool.submit(_count_trudvsem_vacancies, t, region_oktmo): t
                        for t in job_titles
                    }
                    for future in as_completed(future_to_title):
                        title = future_to_title[future]
                        try:
                            counts[title] = future.result()
                        except Exception:
                            counts[title] = -1

                for job_title in job_titles:
                    job_url = (
                        'https://hh.ru/search/vacancy?text=' + _urlencode(job_title)
                        + ('&area=' + area_id if area_id else '')
                    )
                    job_links.append({
                        'title': job_title,
                        'category': _classify_job(job_title),
                        'url': job_url,
                        'hh_count': counts.get(job_title, -1),
                    })

        total_hh_count = sum(j['hh_count'] for j in job_links if j['hh_count'] >= 0)

        keywords = _direction_to_keywords(payload.direction)
        search_url = (
            'https://hh.ru/search/vacancy?text=' + _urlencode(keywords)
            + ('&area=' + area_id if area_id else '')
        )

        dir_cat_label = _get_direction_category(payload.direction)

        return {
            'area_name': area_name,
            'area_id': area_id,
            'dir_category': dir_cat_label,
            'local_count': local_count,
            'job_links': job_links,
            'total_hh_count': total_hh_count,
            'keywords': keywords,
            'search_url': search_url,
        }
    except Exception as e:
        return {'error': str(e)}


