const babyNames = [
    {
      letter: 'A',
      image: "/img/gif/baby- (1)",
      boys: ['Ahmad', 'Ali', 'Amir', 'Ayman', 'Anas', 'Aziz', 'Abdullah', 'Arafat', 'Adnan', 'Ashraf', 'Akram', 'Aqil', 'Arif', 'Ammar'],
      girls: ['Amina', 'Asma', 'Aisha', 'Amal', 'Aya', 'Abeer', 'Ahlam', 'Afaf', 'Anwar', 'Aziza', 'Arwa', 'Amani', 'Aseel', 'Alya']
    },
    {
      letter: 'B',
      image: "/img/gif/baby- (2)",
      boys: ['Bilal', 'Bashir', 'Badr', 'Bassam', 'Burhan', 'Baha', 'Barakat', 'Basil', 'Bahjat', 'Bakr', 'Basel', 'Bashar', 'Burayd', 'Bishr'],
      girls: ['Basma', 'Buthaina', 'Badia', 'Bahira', 'Bushra', 'Balqis', 'Barira', 'Bayan', 'Basira', 'Baraa', 'Batool', 'Banan', 'Basmah', 'Bilqis']
    },
    {
      letter: 'C',
      image: "/img/gif/baby- (3)",
      boys: ['Cairo', 'Chakir', 'Chafik', 'Chahid', 'Chakib', 'Chems', 'Cherif', 'Chaaban', 'Chakir', 'Chawki', 'Chams', 'Chehrazad', 'Chayeb', 'Chawdar'],
      girls: ['Chaima', 'Chiraz', 'Chayma', 'Chahla', 'Chadia', 'Chafika', 'Chahira', 'Chahida', 'Chamsa', 'Chaymaa', 'Chikha', 'Chehrazad', 'Chakiba', 'Chaimaa']
    },
    {
      letter: 'D',
      image: "/img/gif/baby- (4)",
      boys: ['Dawood', 'Dani', 'Dhia', 'Dihya', 'Diyan', 'Dinar', 'Dahi', 'Dahiq', 'Dujan', 'Diyaa', 'Dhafer', 'Dhaif', 'Dakhil', 'Dawid'],
      girls: ['Dina', 'Dalal', 'Dalia', 'Duaa', 'Danah', 'Dareen', 'Diyana', 'Durra', 'Dorra', 'Dima', 'Dunya', 'Daris', 'Dahlia', 'Dafna']
    },
    {
      letter: 'E',
      image: "/img/gif/baby- (5)",
      boys: ['Elias', 'Ehab', 'Ezz', 'Emad', 'Ebrahim', 'Eyman', 'Ehsan', 'Essam', 'Eid', 'Ezzat', 'Elwan', 'Ehabuddin', 'Elham', 'Eidah'],
      girls: ['Eman', 'Ebtisam', 'Elham', 'Esraa', 'Ehsan', 'Eiman', 'Enaam', 'Esmat', 'Elaf', 'Ebtihal', 'Eba', 'Eshraq', 'Ezdihar', 'Enas']
    },
    {
      letter: 'F',
      image: "/img/gif/baby- (6)",
      boys: ['Fahd', 'Faris', 'Fadi', 'Fadel', 'Fawwaz', 'Fahim', 'Farouq', 'Fikri', 'Firas', 'Fuad', 'Fath', 'Fadil', 'Fateh', 'Fidyan'],
      girls: ['Fatima', 'Farah', 'Fayza', 'Ferial', 'Fadia', 'Fadwa', 'Fahima', 'Fairouz', 'Fatin', 'Fiza', 'Furoozan', 'Fakhera', 'Fayrouz', 'Farhana']
    },
    {
      letter: 'G',
      image: "/img/gif/baby- (7)",
      boys: ['Gamal', 'Ghassan', 'Ghanim', 'Ghazi', 'Ghaith', 'Gharib', 'Ghadir', 'Ghani', 'Ghufran', 'Ghumari', 'Ghazali', 'Ghalib', 'Ghufayrah', 'Ghiyath'],
      girls: ['Ghada', 'Ghania', 'Ghaida', 'Ghina', 'Ghulay', 'Ghazal', 'Gulsun', 'Gulzar', 'Ghasna', 'Ghurra', 'Ghinwa', 'Ghaziyah', 'Ghaffari', 'Ghaibah']
    },
    {
      letter: 'H',
      image: "/img/gif/baby- (8)",
      boys: ['Hassan', 'Hussein', 'Hamza', 'Harith', 'Hadi', 'Haitham', 'Hazem', 'Hilal', 'Hatem', 'Hisham', 'Hafiz', 'Humam', 'Habib', 'Harun'],
      girls: ['Hala', 'Hanan', 'Hafsa', 'Huda', 'Husna', 'Hajer', 'Hibah', 'Hayat', 'Hasna', 'Hana', 'Hanadi', 'Hameeda', 'Hamida', 'Hadeel']
    },
    {
      letter: 'I',
      image: "/img/gif/baby- (9)",
      boys: ['Ibrahim', 'Ismail', 'Imran', 'Ilyas', 'Idris', 'Izz', 'Ihsan', 'Ishaq', 'Irfan', 'Issa', 'Ibtisam', 'Imad', 'Ibad', 'Iqbal'],
      girls: ['Iman', 'Iqra', 'Ibtisam', 'Isra', 'Ilham', 'Izzah', 'Inaam', 'Ibtihaal', 'Israa', 'Iffat', 'Izdihar', 'Imtithal', 'Inaya', 'Iqlima']
    },
    {
      letter: 'J',
      image: "/img/gif/baby- (10)",
      boys: ['Jamal', 'Jaber', 'Jalal', 'Jibril', 'Jad', 'Jasim', 'Juhayl', 'Jalil', 'Jumah', 'Jundub', 'Jabir', 'Jafar', 'Jawad', 'Jalil'],
      girls: ['Jumana', 'Jannah', 'Jihan', 'Jamila', 'Joud', 'Jawahir', 'Jalila', 'Jinan', 'Jasmin', 'Jannah', 'Jumayma', 'Juwairiya', 'Jazia', 'Jumaymah']
    },
    {
      letter: 'K',
      image: "/img/gif/baby- (11)",
      boys: ['Kamal', 'Karim', 'Khalid', 'Khayr', 'Khalaf', 'Kaamil', 'Kareem', 'Khaleel', 'Khattab', 'Kinan', 'Kadir', 'Kais', 'Kazim', 'Kamaluddin'],
      girls: ['Khadija', 'Karima', 'Kawthar', 'Kanza', 'Kifaya', 'Kulthum', 'Khadra', 'Khalida', 'Kashifa', 'Khansa', 'Karmila', 'Kiswah', 'Kainat', 'Kamilah']
    },
    {
      letter: 'L',
      image: "/img/gif/baby- (12)",
      boys: ['Luqman', 'Latif', 'Lahab', 'Labib', 'Lazim', 'Labeeb', 'Labeed', 'Lafi', 'Luqaim', 'Luhaym', 'Lutfi', 'Lamis', 'Lubab', 'Lut'],
      girls: ['Laila', 'Lubna', 'Lina', 'Lamis', 'Lamya', 'Latifa', 'Lujain', 'Lama', 'Lina', 'Layal', 'Layla', 'Lubaba', 'Lulwa', 'Layan']
    },
    {
      letter: 'M',
      image: "/img/gif/baby- (13)",
      boys: ['Mohammad', 'Mahmoud', 'Mustafa', 'Musa', 'Mazin', 'Mansour', 'Mujahid', 'Muadh', 'Mikail', 'Malik', 'Mujtaba', 'Murad', 'Mudar', 'Muthana'],
      girls: ['Maryam', 'Maha', 'Mona', 'Mariam', 'Manal', 'Maysa', 'Mehnaz', 'Mahira', 'Maysoon', 'Marwa', 'Majida', 'Masuma', 'Mufida', 'Mahjabeen']
    },
    {
      letter: 'N',
      image: "/img/gif/baby- (14)",
      boys: ['Nabil', 'Naseem', 'Nasser', 'Nuh', 'Naeem', 'Najeeb', 'Najm', 'Nasir', 'Nawfal', 'Nizam', 'Numan', 'Nazir', 'Nadir', 'Nawaz'],
      girls: ['Nadia', 'Nawal', 'Nour', 'Najwa', 'Naima', 'Nasreen', 'Nargis', 'Nashita', 'Nazia', 'Naznin', 'Nihal', 'Nayla', 'Nusayba', 'Nouran']
    },
    {
      letter: 'O',
      image: "/img/gif/baby- (15)",
      boys: ['Omar', 'Osman', 'Owais', 'Othman', 'Obaid', 'Omaran', 'Omari', 'Owaisi', 'Othaim', 'Omid', 'Okeil', 'Owaiz', 'Othaimin', 'Owaib'],
      girls: ['Omneya', 'Olfat', 'Ohood', 'Ojala', 'Oniyah', 'Omera', 'Omdah', 'Omaimah', 'Owaisha', 'Owaida', 'Ozza', 'Oyshee', 'Olamina', 'Omyra']
    },
    {
      letter: 'P',
      image: "/img/gif/baby- (16)",
      boys: ['Parvez', 'Pakeez', 'Parsa', 'Paiman', 'Pirzada', 'Payan', 'Pazeer', 'Paymon', 'Pakiza', 'Parham', 'Patman', 'Pazir', 'Pirmohammed', 'Pirooz'],
      girls: ['Parveen', 'Parisa', 'Pakiza', 'Peyman', 'Poonam', 'Pari', 'Parwana', 'Palwasha', 'Pazira', 'Parwina', 'Parya', 'Pahlaj', 'Peyvand', 'Paimona']
    },
    {
      letter: 'Q',
      image: "/img/gif/baby- (17)",
      boys: ['Qasim', 'Qais', 'Qadir', 'Qudama', 'Qurban', 'Qudsi', 'Qudrat', 'Qutaiba', 'Qusay', 'Qaisar', 'Qamar', 'Qutayba', 'Qabir', 'Qudamah'],
      girls: ['Qamar', 'Qistina', 'Qudsia', 'Qazala', 'Qurrat', 'Qurratulain', 'Qindeel', 'Qasira', 'Qudsiya', 'Qanita', 'Qaniyah', 'Qila', 'Qimat', 'Quds']
    },
    {
      letter: 'R',
      image: "/img/gif/baby- (18)",
      boys: ['Rashid', 'Rami', 'Rauf', 'Rizwan', 'Raheem', 'Raif', 'Rafeeq', 'Rauf', 'Rayyan', 'Rafid', 'Ramzi', 'Raees', 'Rida', 'Rushd'],
      girls: ['Rania', 'Ruqayya', 'Rimsha', 'Rabia', 'Rasha', 'Rabab', 'Rahila', 'Ramsha', 'Reem', 'Rukhsana', 'Rida', 'Raheela', 'Rana', 'Riham']
    },
    {
      letter: 'S',
      image: "/img/gif/baby- (19)",
      boys: ['Salman', 'Saif', 'Sami', 'Sufyan', 'Saqib', 'Sharif', 'Shamil', 'Shakir', 'Sulayman', 'Shazil', 'Siddiq', 'Shihab', 'Shuja', 'Sahir'],
      girls: ['Sara', 'Sumaya', 'Saba', 'Safiya', 'Sahar', 'Samiya', 'Sawsan', 'Suhaila', 'Shaima', 'Sakina', 'Saba', 'Shazia', 'Samira', 'Sundus']
    },
    {
      letter: 'T',
      image: "/img/gif/baby- (20)",
      boys: ['Tariq', 'Tameem', 'Talha', 'Taher', 'Taqi', 'Tawfiq', 'Tawhid', 'Tarique', 'Tufail', 'Taslim', 'Taimoor', 'Tanzeel', 'Tafseer', 'Tahseen'],
      girls: ['Tasneem', 'Tahira', 'Tazeen', 'Tuba', 'Taqwa', 'Tania', 'Tayba', 'Tabinda', 'Talha', 'Tajwar', 'Tahseen', 'Taseen', 'Taslima', 'Tawhida']
    },
    {
      letter: 'U',
      image: "/img/gif/baby- (21)",
      boys: ['Usman', 'Umar', 'Ubaid', 'Uzair', 'Ulfat', 'Umair', 'Uqbah', 'Usaamah', 'Ubaydullah', 'Uwaymir', 'Uzziah', 'Urooj', 'Uthman', 'Umaarah'],
      girls: ['Umme', 'Uzma', 'Ulya', 'Urfa', 'Umaima', 'Ushna', 'Ulfah', 'Urooj', 'Uzuri', 'Uleema', 'Usha', 'Ummarah', 'Umaiza', 'Umnia']
    },
    {
      letter: 'V',
      image: "/img/gif/baby- (22)",
      boys: ['Vaseem', 'Vahid', 'Vasim', 'Vaseer', 'Vayid', 'Vali', 'Vashir', 'Vikram', 'Vaqas', 'Vohid', 'Vashiruddin', 'Vakeel', 'Vikrant', 'Vaseeq'],
      girls: ['Vaniya', 'Vafa', 'Varda', 'Vashni', 'Viyana', 'Vayda', 'Vakeela', 'Venisa', 'Vishra', 'Vahida', 'Vione', 'Vanea', 'Vishma', 'Vanya']
    },
    {
      letter: 'W',
      image: "/img/gif/baby- (23)",
      boys: ['Waseem', 'Wajid', 'Wali', 'Waheed', 'Wahab', 'Wahaj', 'Waseeq', 'Wasif', 'Waqas', 'Waqar', 'Warda', 'Wajahat', 'Wasim', 'Wasil'],
      girls: ['Warda', 'Wafa', 'Widad', 'Waniya', 'Wahiba', 'Wajeeha', 'Wurood', 'Widad', 'Wasila', 'Wahiya', 'Walida', 'Waliyah', 'Waniya', 'Wahiba']
    },
    {
      letter: 'X',
      image: "/img/gif/baby- (24)",
      boys: ['Xavier', 'Xahid', 'Xabeer', 'Xameer', 'Xahir', 'Xayyan', 'Xayad', 'Xabeel', 'Xaydan', 'Xasim', 'Xazier', 'Xali', 'Xanan', 'Xanif'],
      girls: ['Xara', 'Xaina', 'Xamira', 'Xena', 'Xylia', 'Xanira', 'Xamira', 'Xarina', 'Xamara', 'Xanisa', 'Xelina', 'Xarifa', 'Xalia', 'Xerina']
    },
    {
      letter: 'Y',
      image: "/img/gif/baby- (25)",
      boys: ['Yusuf', 'Yasir', 'Younis', 'Yazeed', 'Yaqoub', 'Yahya', 'Yahyan', 'Yazeer', 'Yameen', 'Yassir', 'Yunus', 'Yazdan', 'Yashar', 'Yamun'],
      girls: ['Yasmin', 'Yumna', 'Yusra', 'Yara', 'Yamila', 'Yasira', 'Yaqoota', 'Yasmeena', 'Yuhaina', 'Yazmeen', 'Yumnah', 'Yasrina', 'Yaria', 'Yusmina']
    },
    {
      letter: 'Z',
      image: "/img/gif/baby- (26)",
      boys: ['Zayd', 'Zahir', 'Zaki', 'Zubair', 'Zain', 'Zaheer', 'Zayyan', 'Zafar', 'Zakaria', 'Zeeshan', 'Zameer', 'Zaydan', 'Zareef', 'Zaman'],
      girls: ['Zainab', 'Zara', 'Zoya', 'Zohra', 'Zakia', 'Zarmina', 'Zuleikha', 'Zeba', 'Zayna', 'Zahira', 'Zareen', 'Zubeda', 'Zafira', 'Zeenat']
    }
  ];

module.exports = babyNames;