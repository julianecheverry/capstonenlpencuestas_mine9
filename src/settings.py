# Settings

# Accents to remove
accents = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',

            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'Ä': 'A', 'Ë': 'E', 'Ï': 'I', 'Ö': 'O', 'Ü': 'U',
            'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
            'Â': 'A', 'Ê': 'E', 'Î': 'I', 'Ô': 'O', 'Û': 'U'}

# Stopwords to remove
# Global level

stopwords_global = ['nr', 'académico', 'academia', 'universidad', 'facultad',
                   'ademas', 'tambien', 'ma', 'embargo', 'muy',
                   'externado', 'nop', 'verdad',
                    'nada', 'ninguna', 'ninguno', 'ningun',
                    'no tengo', '1010', '10/10',
                    ':)', ':D', ':p', ':b',
                    'por favor'
                   ]

# Per specific survey
stopwords_evaluaciondocente = ['evaluacion', 'docente', 'docentes', 'profe'
                               'profesor', 'profesora',
                               'profesores', 'profesoras',
                               'persona', 'destaco', 'resalto',
                               'sugerencia', 'recomendacion',
                               'sugiero', 'recomiendo',
                               'oportunidad de mejora',
                               'oportunidades de mejora'
                               ]

stopwords_autoevaluaciondocente = ['resultados aprendizaje',
                                    'percepcion', 'progreso',
                                    'resalto', 'resaltar', 'docente',
                                    'oportunidad de mejora',
                                    'oportunidades de mejora'
                                    # 'estudiantes'
                                ]

stopwords_calidaddocentes = ['sugerencia', 'recomendacion',
                            'sugiero', 'recomiendo',
                            'oportunidad de mejora',
                            'oportunidades de mejora',
                            'resalto', 'resaltar',
                            # 'servicio'
                            ]

stopwords_calidadadministrativos = ['sugerencia', 'recomendacion',
                                    'sugiero', 'recomiendo',
                                    'oportunidad de mejora',
                                    'oportunidades de mejora',
                                    'resalto', 'resaltar',
                                    'deberian',
                                    # 'servicio'
                                    ]

stopwords_calidadestudiantes = ['sugerencia', 'recomendacion',
                                'sugiero', 'recomiendo',
                                'oportunidad de mejora',
                                'oportunidades de mejora',
                                'resalto', 'resaltar',
                                'deberian',
                                # 'servicio'
                                ]

stopwords_calidaddirectivos = [ 'sugerencia', 'recomendacion',
                                'sugiero', 'recomiendo',
                                'oportunidad de mejora',
                                'oportunidades de mejora',
                                'resalto', 'resaltar',
                                'deberian',
                                # 'servicio'
                                ]

stopwords_calidadegresados = [ 'sugerencia', 'recomendacion',
                                'sugiero', 'recomiendo',
                                'oportunidad de mejora',
                                'oportunidades de mejora',
                                'resalto', 'resaltar',
                                'deberian',
                                # 'servicio'
                                ]