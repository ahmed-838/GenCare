// Swiper initialization function
function initSwiper() {
  // Destroy existing swiper instance if it exists
  if (window.cardSwiper) {
    window.cardSwiper.destroy(true, true);
  }
  
  // Initialize new swiper
  window.cardSwiper = new Swiper('.card-wrapper', {
    loop: true,
    spaceBetween: 30,
    pagination: {
      el: '.swiper-pagination',
    },
  
    // Navigation arrows
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
  
    breakpoints: {
      0: {
        slidesPerView: 1
      },
      768: {
        slidesPerView: 2
      },
      1024: {
        slidesPerView: 3
      },
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  // Initialize language state from localStorage if available
  isArabic = localStorage.getItem('isArabic') === 'true';
  
  // Set the initial language button text
  if (isArabic) {
    langToggle.querySelector('.toggle-lang').textContent = 'العربية';
  } else {
    langToggle.querySelector('.toggle-lang').textContent = 'English';
  }
  
  // Initialize dark mode from localStorage if available
  const isDarkMode = localStorage.getItem('darkMode') === 'true';
  if (isDarkMode) {
    document.body.classList.add('dark-mode');
    darkModeToggle.innerHTML='<span class="toggle-on"><i class="fa-solid fa-lightbulb"></i></span>';
  }

  const diseases = [
      {
        img: '/img/Diseases/Arachnoid_Cyst_Treated_using_a_VP_Shunt.png',
        badge: '1th - 12th Weeks',
        title: 'Arachnoid Cyst',
        description: "is a fluid-filled sac that usually forms in the brain or spinal cord, within the arachnoid membrane surrounding the central nervous system. Often, it is congenital (present from birth), but it can also develop due to brain injury or infection.<br><br>" +
                    "<strong>Symptoms</strong><br>Most arachnoid cysts are asymptomatic and are discovered incidentally. However, if the cyst is large or exerts pressure on brain tissue, it may lead to:<br>" +
                    "- Persistent headache<br>" +
                    "- Dizziness or balance issues<br>" +
                    "- Visual or hearing impairment<br>" +
                    "- Concentration difficulties or memory issues<br><br>" +
                    "<strong>Treatment</strong><br>" +
                    "- <em>Monitoring</em>: If the cyst does not cause symptoms, doctors may prefer regular monitoring.<br>" +
                    "- <em>Surgical Intervention</em>: If symptoms significantly impact the patient’s quality of life, surgery may be performed to drain the cyst fluid or partially or fully remove it to relieve pressure on the brain.<br><br>" +
                    "<strong>Awareness</strong><br>" +
                    "- Early diagnosis is important, especially if symptoms appear.<br>" +
                    "- Raising awareness that most arachnoid cysts are benign and often harmless, though they may need regular check-ups.<br>" +
                    "- Guiding individuals with symptoms to seek specialist help if they experience unusual symptoms like persistent headaches or balance problems.",        badge_ar: 'الأسبوع 1 - 12',
        title_ar: 'كيسة عنكبوتية',
        description_ar: "هي كيس مملوء بالسوائل يتكوّن عادة في الدماغ أو الحبل الشوكي، داخل غشاء العنكبوتية الذي يحيط بالجهاز العصبي المركزي. غالبًا ما تكون خلقية (موجودة منذ الولادة)، ولكن يمكن أن تتكوّن أيضًا نتيجة إصابة دماغية أو عدوى.<br><br>" +
               "<strong>الأعراض</strong><br>معظم الكيسات العنكبوتية لا تُسبب أعراضًا وتُكتشف بالصدفة. ولكن، إذا كانت الكيسة كبيرة أو تضغط على أنسجة الدماغ، فقد تؤدي إلى:<br>" +
               "- صداع مستمر<br>" +
               "- دوخة أو مشاكل في التوازن<br>" +
               "- ضعف في الرؤية أو السمع<br>" +
               "- صعوبة في التركيز أو مشاكل في الذاكرة<br><br>" +
               "<strong>العلاج</strong><br>" +
               "- <em>المراقبة</em>: إذا لم تُسبب الكيسة أعراضًا، فقد يفضل الأطباء مراقبتها بانتظام.<br>" +
               "- <em>التدخل الجراحي</em>: إذا أثرت الأعراض بشكل كبير على جودة حياة المريض، قد تُجرى جراحة لتصريف السائل أو إزالة الكيسة جزئيًا أو كليًا لتخفيف الضغط على الدماغ.<br><br>" +
               "<strong>التوعية</strong><br>" +
               "- التشخيص المبكر مهم، خاصة إذا ظهرت أعراض.<br>" +
               "- رفع الوعي بأن معظم الكيسات العنكبوتية حميدة وغالبًا لا تُسبب ضررًا، لكنها قد تحتاج لمتابعة منتظمة.<br>" +
               "- توجيه الأفراد الذين يعانون من أعراض غير معتادة مثل الصداع المستمر أو مشاكل التوازن إلى مراجعة أخصائي."

      },
      {
        img: '/img/Diseases/Clinical-impression-of-siblings-of-family-1-a-patient-1-b-patient-2.png',
        badge: '1th - 12th Weeks',
        title: 'Cerebellar Hypoplasia',
        description: "is a condition where the cerebellum (the part of the brain responsible for coordination, balance, and motor control) does not develop properly. It can be <strong>congenital</strong> (present at birth) or <strong>acquired</strong> (due to brain damage after birth).<br>"+
              "<strong>Causes:</strong><br>- <strong>Congenital:</strong> Genetic disorders, prenatal infections (e.g., rubella), or lack of oxygen during pregnancy.<br>- <strong>Acquired:</strong> Brain damage due to lack of oxygen at birth, trauma, or infections like meningitis.<br>"+
              "<strong>Symptoms:</strong><br>- <strong>Coordination issues</strong> (ataxia), difficulty walking, and muscle weakness.<br>- <strong>Speech problems</strong> (slurred speech).<br>- <strong>Balance difficulties</strong> and possible <strong>seizures</strong>.<br>- <strong>Cognitive delays</strong> or developmental problems in some cases.<br>"+
              "<strong>Diagnosis:</strong><br>- <strong>MRI scans</strong> to visualize the cerebellum and assess its development.<br>- <strong>Genetic testing</strong> to identify underlying genetic causes.<br>- <strong>Neurological exams</strong> to assess coordination and motor function.<br>"+
              "<strong>Treatment:</strong><br>- <strong>Physical Therapy</strong> to improve coordination, balance, and strength.<br>- <strong>Occupational Therapy</strong> to help with daily tasks.<br>- <strong>Speech Therapy</strong> if speech is affected.<br>- <strong>Medications</strong> to manage symptoms like muscle spasms or seizures.<br>- <strong>Surgical interventions</strong> are rare but may be needed for certain complications.<br>"+
              "<strong>Prevention:</strong><br>- <strong>Good prenatal care</strong> to monitor fetal development.<br>- <strong>Vaccinations</strong> (e.g., rubella) before pregnancy to avoid infections.<br>- <strong>Genetic counseling</strong> for families with a history of genetic disorders.<br>"+
              "<strong>Prognosis:</strong> The outcome depends on the severity of the condition. Some individuals may lead independent lives with therapy, while others may need lifelong support for mobility and daily activities.<br>"+
              "In summary, <strong>Cerebellar Hypoplasia</strong> is a neurological condition affecting coordination and balance, and while there is no cure, early treatment with therapy and support can help improve quality of life.",        badge_ar: 'الأسبوع 1 - 12',
        title_ar: 'نقص تنسج المخيخ',
  
        description_ar: "هو حالة لا يتطور فيها المخيخ (الجزء من الدماغ المسؤول عن التنسيق، التوازن، والتحكم الحركي) بشكل طبيعي. يمكن أن تكون <strong>خلقية</strong> (موجودة منذ الولادة) أو <strong>مكتسبة</strong> (نتيجة تلف في الدماغ بعد الولادة).<br>" +
            "<strong>الأسباب:</strong><br>- <strong>خلقية:</strong> اضطرابات وراثية، التهابات أثناء الحمل (مثل الحصبة الألمانية)، أو نقص الأكسجين خلال الحمل.<br>- <strong>مكتسبة:</strong> تلف في الدماغ بسبب نقص الأكسجين عند الولادة، إصابات، أو التهابات مثل التهاب السحايا.<br>" +
            "<strong>الأعراض:</strong><br>- <strong>مشاكل في التنسيق</strong> (ترنّح)، صعوبة في المشي، وضعف في العضلات.<br>- <strong>مشاكل في النطق</strong> (كلام غير واضح).<br>- <strong>صعوبات في التوازن</strong> واحتمال حدوث <strong>نوبات صرع</strong>.<br>- <strong>تأخر إدراكي</strong> أو مشاكل في النمو في بعض الحالات.<br>" +
            "<strong>التشخيص:</strong><br>- <strong>تصوير بالرنين المغناطيسي (MRI)</strong> لتصوير المخيخ وتقييم تطوره.<br>- <strong>اختبارات جينية</strong> لتحديد الأسباب الوراثية المحتملة.<br>- <strong>فحوصات عصبية</strong> لتقييم التنسيق والوظائف الحركية.<br>" +
            "<strong>العلاج:</strong><br>- <strong>العلاج الطبيعي</strong> لتحسين التنسيق، التوازن، والقوة.<br>- <strong>العلاج الوظيفي</strong> للمساعدة في أداء المهام اليومية.<br>- <strong>علاج النطق</strong> إذا تأثر الكلام.<br>- <strong>الأدوية</strong> للتحكم في الأعراض مثل التشنجات العضلية أو نوبات الصرع.<br>- <strong>التدخل الجراحي</strong> نادر ولكن قد يكون ضروريًا لبعض المضاعفات.<br>" +
            "<strong>الوقاية:</strong><br>- <strong>رعاية ما قبل الولادة الجيدة</strong> لمتابعة تطور الجنين.<br>- <strong>التطعيمات</strong> (مثل الحصبة الألمانية) قبل الحمل لتجنب العدوى.<br>- <strong>الاستشارة الوراثية</strong> للعائلات التي لديها تاريخ من الاضطرابات الوراثية.<br>" +
            "<strong>التوقعات:</strong> تعتمد النتيجة على شدة الحالة. بعض الأفراد قد يعيشون حياة مستقلة بمساعدة العلاج، بينما قد يحتاج آخرون إلى دعم مدى الحياة في الحركة والأنشطة اليومية.<br>" +
            "في الخلاصة، <strong>نقص تنسج المخيخ</strong> هو حالة عصبية تؤثر على التنسيق والتوازن، وعلى الرغم من عدم وجود علاج شافٍ، فإن العلاج المبكر والدعم يمكن أن يساعد في تحسين جودة الحياة."  

                    },
      {
        img: '/img/Diseases/767679ab89a64fa5285bea4d60f36dce.jpg',
        badge: '1th - 26th Weeks',
        title: 'Colphocephaly',
        description: "Colpocephaly is a rare congenital brain abnormality characterized by the disproportionate enlargement " +
              "of the occipital horns of the brain's lateral ventricles. This occurs due to abnormal brain development, " +
              "often associated with issues in the white matter.<br>" +
              "<br>" +
              "<strong>Causes:</strong><br>" +
              "- <strong>Developmental Brain Abnormalities:</strong> Problems in neuronal migration during fetal development.<br>" +
              "- <strong>Associated Conditions:</strong> Corpus callosum agenesis or hypoplasia (underdevelopment or absence of the corpus callosum).<br>" +
              "- <strong>Genetic or Environmental Factors:</strong> Chromosomal abnormalities, prenatal infections, or injuries.<br>" +
              "<br>" +
              "<strong>Symptoms:</strong><br>" +
              "Varies depending on severity; common symptoms include:<br>" +
              "- <strong>Developmental delays:</strong> Slower milestones in motor and cognitive skills.<br>" +
              "- <strong>Seizures:</strong> Common in severe cases.<br>" +
              "- <strong>Intellectual disabilities:</strong> Ranging from mild to severe.<br>" +
              "- <strong>Muscle weakness:</strong> Spasticity or lack of coordination.<br>" +
              "<br>" +
              "<strong>Diagnosis:</strong><br>" +
              "- <strong>Imaging Studies:</strong> MRI or CT scans show enlarged occipital horns and reduced white matter.<br>" +
              "- <strong>Prenatal Ultrasound:</strong> Sometimes identifies abnormalities during pregnancy.<br>" +
              "- <strong>Genetic Testing:</strong> To investigate potential chromosomal issues.<br>" +
              "<br>" +
              "<strong>Treatment:</strong><br>" +
              "No cure; management focuses on symptoms:<br>" +
              "- <strong>Physical therapy:</strong> For motor skills.<br>" +
              "- <strong>Medications:</strong> For seizures.<br>" +
              "- <strong>Special education:</strong> To address learning disabilities.<br>" +
              "- <strong>Speech therapy:</strong> For communication difficulties.<br>" +
              "<br>" +
              "<strong>Prognosis:</strong><br>" +
              "Varies by severity:<br>" +
              "- Mild cases may lead relatively normal lives with support.<br>" +
              "- Severe cases may require lifelong care and face significant neurological challenges.",        badge_ar: 'الأسبوع 1 - 26',
        title_ar: 'كولفوسيفالي',
        description_ar: " هو خلل نادر في الدماغ منذ الولادة يتميز بالتوسّع غير المتناسب " +
              "للقرون القذالية للبطينين الجانبيين في الدماغ. يحدث هذا نتيجة لتطور غير طبيعي في الدماغ، " +
              "وغالبًا ما يكون مرتبطًا بمشكلات في المادة البيضاء.<br>" +
              "<br>" +
              "<strong>الأسباب:</strong><br>" +
              "- <strong>تشوهات في تطور الدماغ:</strong> مشاكل في هجرة الخلايا العصبية أثناء تطور الجنين.<br>" +
              "- <strong>حالات مرتبطة:</strong> غياب الجسم الثفني أو نقص نموه (تطور غير مكتمل أو غيابه تمامًا).<br>" +
              "- <strong>عوامل وراثية أو بيئية:</strong> اضطرابات صبغية، التهابات أثناء الحمل، أو إصابات.<br>" +
              "<br>" +
              "<strong>الأعراض:</strong><br>" +
              "تختلف حسب شدة الحالة؛ تشمل الأعراض الشائعة:<br>" +
              "- <strong>تأخر في النمو:</strong> بطء في الوصول إلى المهارات الحركية والمعرفية.<br>" +
              "- <strong>نوبات صرع:</strong> شائعة في الحالات الشديدة.<br>" +
              "- <strong>إعاقات ذهنية:</strong> تتراوح من خفيفة إلى شديدة.<br>" +
              "- <strong>ضعف العضلات:</strong> التشنج أو ضعف التنسيق.<br>" +
              "<br>" +
              "<strong>التشخيص:</strong><br>" +
              "- <strong>الفحوصات التصويرية:</strong> يُظهر التصوير بالرنين المغناطيسي أو الأشعة المقطعية توسع القُرون القذالية ونقص في المادة البيضاء.<br>" +
              "- <strong>الموجات فوق الصوتية قبل الولادة:</strong> قد تُكتشف أحيانًا أثناء الحمل.<br>" +
              "- <strong>الاختبارات الجينية:</strong> للتحقق من وجود اضطرابات صبغية محتملة.<br>" +
              "<br>" +
              "<strong>العلاج:</strong><br>" +
              "لا يوجد علاج شافٍ؛ يركّز التدخل على إدارة الأعراض:<br>" +
              "- <strong>العلاج الطبيعي:</strong> لتحسين المهارات الحركية.<br>" +
              "- <strong>الأدوية:</strong> للسيطرة على نوبات الصرع.<br>" +
              "- <strong>التعليم الخاص:</strong> لمعالجة صعوبات التعلم.<br>" +
              "- <strong>علاج النطق:</strong> لتحسين مهارات التواصل.<br>" +
              "<br>" +
              "<strong>التوقعات:</strong><br>" +
              "تختلف حسب الشدة:<br>" +
              "- قد يعيش الأفراد في الحالات الخفيفة حياة شبه طبيعية مع الدعم.<br>" +
              "- قد تتطلب الحالات الشديدة رعاية مدى الحياة وتواجه تحديات عصبية كبيرة."

      },
      {
        img: '/img/Diseases/WhatsApp Image 2025-01-09 at 18.00.51_fed83f97.jpg',
        badge: '3th - 4th Months',
        title: 'Encephalocele',
        description: "Encephalocele is a rare congenital condition where brain tissue and membranes protrude through an opening " +
              "in the skull. It is a type of neural tube defect that occurs during early fetal development when the skull " +
              "bones do not close completely.<br>" +
              "<br>" +
              "<strong>Causes:</strong><br>" +
              "- <strong>Genetic Factors:</strong> Family history of neural tube defects.<br>" +
              "- <strong>Environmental Factors:</strong> Maternal folic acid deficiency. Exposure to certain toxins or medications during pregnancy.<br>" +
              "- <strong>Associated Syndromes:</strong> Often linked to other genetic conditions or syndromes.<br>" +
              "<br>" +
              "<strong>Symptoms:</strong><br>" +
              "Visible sac-like protrusion on the head (can be at the back, front, or base of the skull).<br>" +
              "Neurological problems depending on the severity, such as:<br>" +
              "- <strong>Developmental delays:</strong> May be present.<br>" +
              "- <strong>Seizures:</strong> May occur in some cases.<br>" +
              "- <strong>Vision or motor impairments:</strong> Can be seen depending on the severity.<br>" +
              "- <strong>Hydrocephalus:</strong> Fluid buildup in the brain.<br>" +
              "<br>" +
              "<strong>Diagnosis:</strong><br>" +
              "<strong>Prenatal Testing:</strong><br>" +
              "- <strong>Ultrasound:</strong> Detects encephalocele during pregnancy.<br>" +
              "- <strong>Alpha-fetoprotein (AFP) Test:</strong> High levels may indicate neural tube defects.<br>" +
              "<strong>Postnatal Imaging:</strong><br>" +
              "- <strong>MRI or CT scans:</strong> Assess the extent and content of the protrusion.<br>" +
              "<br>" +
              "<strong>Treatment:</strong><br>" +
              "<strong>Surgical Repair:</strong><br>" +
              "- Remove the sac and return brain tissue to the skull.<br>" +
              "- Close the skull defect.<br>" +
              "<strong>Supportive Care:</strong><br>" +
              "- Physical and occupational therapy for developmental support.<br>" +
              "- Seizure management with medications.<br>" +
              "<br>" +
              "<strong>Prognosis:</strong><br>" +
              "Depends on the size, location, and associated complications.<br>" +
              "- Small encephaloceles with no brain tissue involvement may have a good outcome after surgery.<br>" +
              "- Larger or complex cases may lead to significant neurological and developmental challenges.",        badge_ar: 'الشهر 3 - 4',
        title_ar: 'القيلة الدماغية',
        description_ar: " هو حالة خلقية نادرة حيث يبرز نسيج الدماغ والأغشية من خلال فتحة " +
            "في الجمجمة. وهو نوع من عيوب الأنبوب العصبي يحدث أثناء التطور الجنيني المبكر عندما لا تُغلق " +
            "عظام الجمجمة بالكامل.<br>" +
            "<br>" +
            "<strong>الأسباب:</strong><br>" +
            "- <strong>عوامل وراثية:</strong> وجود تاريخ عائلي لعيوب الأنبوب العصبي.<br>" +
            "- <strong>عوامل بيئية:</strong> نقص حمض الفوليك عند الأم، أو التعرض لسموم أو أدوية معينة أثناء الحمل.<br>" +
            "- <strong>متلازمات مرتبطة:</strong> غالبًا ما تكون مرتبطة باضطرابات وراثية أو متلازمات أخرى.<br>" +
            "<br>" +
            "<strong>الأعراض:</strong><br>" +
            "نتوء مرئي يشبه الكيس على الرأس (قد يكون في الخلف، الأمام، أو قاعدة الجمجمة).<br>" +
            "مشاكل عصبية حسب الشدة، مثل:<br>" +
            "- <strong>تأخر في النمو:</strong> قد يكون موجودًا.<br>" +
            "- <strong>نوبات صرع:</strong> قد تحدث في بعض الحالات.<br>" +
            "- <strong>ضعف في الرؤية أو الحركة:</strong> حسب شدة الحالة.<br>" +
            "- <strong>الاستسقاء الدماغي:</strong> تراكم السوائل في الدماغ.<br>" +
            "<br>" +
            "<strong>التشخيص:</strong><br>" +
            "<strong>اختبارات قبل الولادة:</strong><br>" +
            "- <strong>الموجات فوق الصوتية:</strong> تكتشف الحالة أثناء الحمل.<br>" +
            "- <strong>اختبار ألفا فيتو بروتين (AFP):</strong> ارتفاع مستوياته قد يدل على عيوب في الأنبوب العصبي.<br>" +
            "<strong>تصوير بعد الولادة:</strong><br>" +
            "- <strong>الرنين المغناطيسي أو الأشعة المقطعية:</strong> لتقييم حجم ومحتوى البروز.<br>" +
            "<br>" +
            "<strong>العلاج:</strong><br>" +
            "<strong>الإصلاح الجراحي:</strong><br>" +
            "- إزالة الكيس وإعادة نسيج الدماغ إلى الجمجمة.<br>" +
            "- إغلاق فتحة الجمجمة.<br>" +
            "<strong>الرعاية الداعمة:</strong><br>" +
            "- علاج طبيعي ووظيفي لدعم النمو.<br>" +
            "- أدوية للتحكم في نوبات الصرع.<br>" +
            "<br>" +
            "<strong>التوقعات:</strong><br>" +
            "تعتمد على الحجم والموقع والمضاعفات المرتبطة.<br>" +
            "- الحالات الصغيرة بدون تورط أنسجة دماغية قد يكون لها نتائج جيدة بعد الجراحة.<br>" +
            "- الحالات الكبيرة أو المعقدة قد تؤدي إلى تحديات عصبية وتنموية كبيرة."


      },
      {
        img: '/img/Diseases/Ventriculomegaly-Illustration_2021-05-04-152710.png',
        badge: '18th - 24 Weeks',
        title: 'Mild Ventriculomegaly',
        description: "Mild Ventriculomegaly is a condition characterized by a slight enlargement of the brain's ventricles, " +
          "which are fluid-filled spaces containing cerebrospinal fluid (CSF). This condition can be detected through " +
          "prenatal ultrasounds or postnatally via imaging techniques like <strong>MRI</strong> or <strong>CT scans</strong>.<br>" +
          "<br>" +
          "<strong>Causes:</strong><br>" +
          "- <strong>Genetic conditions:</strong> Associated with certain genetic disorders or brain abnormalities.<br>" +
          "- <strong>Developmental brain issues:</strong> May result from abnormal brain development during pregnancy.<br>" +
          "- <strong>Brain injuries or infections:</strong> Brain damage or infections can sometimes lead to ventricular enlargement.<br>" +
          "- <strong>Increased intracranial pressure:</strong> Elevated pressure within the skull can cause the ventricles to expand.<br>" +
          "<br>" +
          "<strong>When It Occurs During Pregnancy:</strong><br>" +
          "Mild ventriculomegaly is often detected during routine ultrasound scans in the second trimester (18th–22nd week). " +
          "It is diagnosed when the ventricles measure between 10–15 millimeters. In many cases, it is a benign finding, " +
          "particularly when no other abnormalities are present.<br>" +
          "<br>" +
          "<strong>Effects and Symptoms:</strong><br>" +
          "- Often asymptomatic and resolves over time.<br>" +
          "- If associated with other conditions, it may cause developmental delays or neurological issues, though this is not common.<br>" +
          "<br>" +
          "<strong>Diagnosis:</strong><br>" +
          "- Typically identified through prenatal ultrasound.<br>" +
          "- Additional imaging (e.g., <strong>MRI</strong>) may be required to monitor progression or resolution.<br>" +
          "<br>" +
          "<strong>Treatment:</strong><br>" +
          "- In most cases, no treatment is necessary, only regular monitoring via follow-up ultrasounds.<br>" +
          "- If linked to an underlying condition (e.g., genetic disorder or brain abnormality), treatment targets the root cause.<br>" +
          "<br>" +
          "<strong>Prognosis:</strong><br>" +
          "Mild ventriculomegaly is often not a cause for concern, and with proper monitoring, pregnancies proceed normally. " +
          "Developmental outcomes are typically positive when the condition resolves on its own or occurs in isolation.",        badge_ar: 'الأسبوع 18 - 24',
        title_ar: 'توسع البطينات الخفيف',
        description_ar: "هو اتساع طفيف في بطينات الدماغ يتميز بحالة من تضخم البطينات المملوءة بالسائل الدماغي النخاعي (CSF). " +
                     "يمكن اكتشاف هذه الحالة مبكرًا من خلال تقنيات التصوير مثل <strong>الموجات فوق الصوتية</strong> أثناء الحمل أو من خلال تقنيات التصوير مثل <strong>الرنين المغناطيسي</strong>.<br>" +
                     "<br>" +
                     "<strong>الأسباب:</strong><br>" +
                     "- <strong>الاضطرابات الوراثية أو تشوهات الدماغ:</strong> مرتبطة ببعض الأسباب الجينية.<br>" +
                     "- <strong>الحمل:</strong> قد تكون نتيجة لتطور غير طبيعي للدماغ أثناء الحمل.<br>" +
                     "- <strong>إصابات أو التهابات دماغية:</strong> يمكن أن تؤدي إلى توسع البطينات أحيانًا.<br>" +
                     "- <strong>زيادة الضغط داخل الجمجمة:</strong> قد يؤدي ارتفاع الضغط فقط إلى توسع البطينات داخل الجمجمة.<br>" +
                     "<br>" +
                     "<strong>متى يحدث أثناء الحمل:</strong><br>" +
                     "يتم تشخيص توسع البطينات الطفيف خلال الفحوصات الروتينية بالموجات فوق الصوتية في الثلث الثاني من الحمل (الأسبوع 18 إلى 22). " +
                     "يتم التشخيص عندما تتراوح قياسات البطينات بين 10–15 ملم، في العديد من الحالات يكون ذلك أمرًا غير مقلق، خصوصًا عند غياب أي تشوهات أخرى.<br>" +
                     "<br>" +
                     "<strong>التأثيرات والأعراض:</strong><br>" +
                     "- غالبًا لا توجد أعراض وقد يختفي بمرور الوقت.<br>" +
                     "- إذا ارتبط بحالات أخرى، فقد يتسبب تأخرًا في النمو أو مشاكل عصبية، لكن هذا غير شائع.<br>" +
                     "<br>" +
                     "<strong>التشخيص:</strong><br>" +
                     "- يتم تحديد الحالة غالبًا من خلال فحص الموجات فوق الصوتية.<br>" +
                     "- يمكن استخدام <strong>MRI</strong> لمتابعة التغيرات أو التأكد من التشخيص.<br>" +
                     "<br>" +
                     "<strong>العلاج:</strong><br>" +
                     "- في معظم الحالات، لا يتطلب علاجًا بل فقط متابعة دورية.<br>" +
                     "- إذا كان مرتبطًا بحالة مرضية أخرى، يُعالج السبب الأساسي.<br>" +
                     "<br>" +
                     "<strong>التوقعات المستقبلية:</strong><br>" +
                     "عادةً ما تكون النتائج إيجابية في حالات Mild Ventriculomegaly، خصوصًا إذا حدثت بشكل منعزل واختفت قبل الولادة أو بعدها بفترة قصيرة."

      },
      {
        img: '/img/Diseases/WhatsApp Image 2025-01-10 at 14.37.12_76b0cb58.jpg',
        badge: '18th - 22th Weeks',
        title: 'Moderate Ventriculomegaly',
        description: "Moderate Ventriculomegaly is a condition where the brain's ventricles are enlarged more than in mild ventriculomegaly " +
          "but not as severely as in more serious cases. The ventricles are fluid-filled spaces that contain cerebrospinal fluid (CSF), " +
          "which helps protect and nourish the brain. Ventricular enlargement can indicate underlying issues with brain development or function.<br>" +
          "<br>" +
          "<strong>Causes:</strong><br>" +
          "- <strong>Genetic disorders:</strong> Conditions like Down syndrome, Edwards syndrome, or other chromosomal abnormalities can lead to moderate ventriculomegaly.<br>" +
          "- <strong>Brain development problems:</strong> Abnormal brain development during pregnancy, such as hydrocephalus or other brain conditions, can cause this condition.<br>" +
          "- <strong>Infections during pregnancy:</strong> Infections like toxoplasmosis or cytomegalovirus (CMV) can result in ventriculomegaly.<br>" +
          "- <strong>Traumatic brain injuries:</strong> Brain damage from trauma or infection may enlarge the ventricles.<br>" +
          "- <strong>Obstructions in cerebrospinal fluid flow:</strong> Blockages like tumors or cysts can cause the ventricles to expand.<br>" +
          "<br>" +
          "<strong>When It Occurs During Pregnancy:</strong><br>" +
          "Moderate ventriculomegaly is usually diagnosed through ultrasound scans between the 18th and 22nd weeks of pregnancy. " +
          "Ventricles measuring between 15 and 20 millimeters are considered enlarged. Detailed anatomical scans often detect this condition.<br>" +
          "<br>" +
          "<strong>Effects and Symptoms:</strong><br>" +
          "- <strong>During pregnancy:</strong> Moderate ventriculomegaly may not show immediate symptoms in the fetus but can signal an underlying issue affecting brain development.<br>" +
          "- <strong>At birth:</strong> The effects depend on the underlying cause. Some children may show no long-term impact, while others could experience developmental delays, cognitive challenges, or motor issues.<br>" +
          "<br>" +
          "<strong>Diagnosis:</strong><br>" +
          "- Detected via prenatal ultrasound.<br>" +
          "- Further imaging like <strong>MRI</strong> may be recommended for a clearer view of the brain and ventricles.<br>" +
          "- Genetic testing (e.g., amniocentesis or CVS) may be performed to check for chromosomal abnormalities.<br>" +
          "<br>" +
          "<strong>Treatment:</strong><br>" +
          "- If caused by hydrocephalus, surgery (e.g., shunt insertion) may be needed to drain excess cerebrospinal fluid.<br>" +
          "- For genetic disorders, specialized care and developmental support may be required after birth.<br>" +
          "- If no major brain abnormalities are present, regular monitoring through follow-up imaging may suffice.<br>" +
          "<br>" +
          "<strong>Prognosis:</strong><br>" +
          "The outcome depends on the underlying cause. Some children may develop normally, while others might face cognitive or physical delays. " +
          "Early detection, close monitoring, and timely intervention can significantly improve the prognosis.<br>" +
          "<br>" +
          "Expecting parents should work closely with their healthcare provider to understand the condition and its implications. Regular follow-ups and " +
          "additional testing are essential to monitor the baby’s development and health.",        badge_ar: 'الأسبوع 18 - 22',
        title_ar: 'توسع البطينات المتوسط',
        description_ar: "تضخم البطينات الدماغية بدرجة متوسطة (Moderate Ventriculomegaly) هو حالة تتضخم فيها بطينات الدماغ (المساحات المملوءة بالسائل) بشكل أكبر من التضخم الخفيف، ولكن ليس بنفس شدة التضخم الشديد.<br>" +
             "البطينات تحتوي على السائل الدماغي الشوكي (CSF) الذي يحمي ويغذي الدماغ. يشير تضخم البطينات إلى احتمال وجود مشاكل في نمو أو وظيفة الدماغ.<br>" +
             "<br>" +
             "<strong>الأسباب:</strong><br>" +
             "- <strong>الاضطرابات الجينية:</strong> مثل متلازمة داون، متلازمة إدواردز، أو شذوذات كروموسومية أخرى.<br>" +
             "- <strong>مشاكل نمو الدماغ:</strong> مثل الاستسقاء الدماغي أو تشوهات أخرى أثناء الحمل.<br>" +
             "- <strong>العدوى داخل الرحم:</strong> مثل داء المقوسات (Toxoplasmosis) أو الفيروس المضخم للخلايا (CMV).<br>" +
             "- <strong>إصابات الدماغ:</strong> تلف ناتج عن صدمات أو التهابات.<br>" +
             "- <strong>انسداد في تدفق السائل الدماغي الشوكي:</strong> مثل وجود أكياس أو أورام تؤدي إلى توسع البطينات.<br>" +
             "<br>" +
             "<strong>وقت التشخيص أثناء الحمل:</strong><br>" +
             "يُشخص عادة بين الأسبوع 18 و22 من الحمل باستخدام الموجات فوق الصوتية. يتم اعتبار البطينات متضخمة عندما يتراوح قطرها بين 15 و20 ملم.<br>" +
             "<br>" +
             "<strong>الأعراض والآثار:</strong><br>" +
             "- <strong>أثناء الحمل:</strong> قد لا تظهر أعراض واضحة، لكن التضخم قد يشير إلى مشاكل نمو الدماغ.<br>" +
             "- <strong>عند الولادة:</strong> تختلف التأثيرات حسب السبب؛ قد لا تظهر أعراض على بعض الأطفال، بينما يعاني آخرون من تأخر نمائي أو مشكلات معرفية أو حركية.<br>" +
             "<br>" +
             "<strong>التشخيص:</strong><br>" +
             "- <strong>الموجات فوق الصوتية قبل الولادة:</strong> للكشف عن التضخم.<br>" +
             "- <strong>الرنين المغناطيسي (MRI):</strong> لتصوير أكثر دقة للدماغ والبطينات.<br>" +
             "- <strong>الاختبارات الجينية:</strong> مثل البزل الأمنيوسي أو أخذ عينات من الزغابات المشيمية (CVS).<br>" +
             "<br>" +
             "<strong>العلاج:</strong><br>" +
             "- <strong>في حال وجود استسقاء دماغي:</strong> قد يلزم إجراء جراحة لتركيب تحويلة لتصريف السائل الزائد.<br>" +
             "- <strong>في حال وجود اضطرابات جينية:</strong> يتم تقديم رعاية متخصصة ودعم نمائي بعد الولادة.<br>" +
             "- <strong>في حال عدم وجود تشوهات كبيرة:</strong> تتم المتابعة المنتظمة عبر التصوير.<br>" +
             "<br>" +
             "<strong>التكهن:</strong><br>" +
             "يعتمد على السبب الكامن. بعض الأطفال يتطورون طبيعيًا، بينما يعاني آخرون من تأخر في النمو أو إعاقات. الاكتشاف المبكر والمراقبة الدقيقة والتدخل المناسب تُحسّن المآل بشكل كبير."


      },
      {
        img: '/img/Diseases/Porencephaly-1.png',
        badge: '8th - 24th Weeks',
        title: 'Porencephaly',
        description:
        "Porencephaly is a rare brain malformation where the cortex develops abnormally, leading to small, irregular folds " +
          "instead of the normal structure. This condition can cause developmental delays, seizures, intellectual disabilities, " +
          "and motor issues.<br>" +
          "<br>" +
          "<strong>Causes:</strong><br>" +
          "- <strong>Genetic Mutations:</strong> (e.g., TUBA1A, DCX).<br>" +
          "- <strong>Infections During Pregnancy:</strong> (e.g., Zika virus, CMV).<br>" +
          "- <strong>Oxygen Deprivation:</strong> during fetal development.<br>" +
          "- <strong>Toxic Exposures:</strong> during pregnancy.<br>" +
          "<br>" +
          "<strong>Symptoms:</strong><br>" +
          "- Developmental delays.<br>" +
          "- Seizures.<br>" +
          "- Muscle weakness or spasticity.<br>" +
          "- Cognitive and speech impairments.<br>" +
          "<br>" +
          "<strong>Diagnosis:</strong><br>" +
          "- <strong>MRI:</strong> for brain imaging.<br>" +
          "- <strong>Genetic Testing:</strong> to identify mutations.<br>" +
          "<br>" +
          "<strong>Treatment:</strong><br>" +
          "- Medications for seizures.<br>" +
          "- Physical, speech, and occupational therapy.<br>" +
          "- Supportive care for mobility and learning challenges.<br>" +
          "<br>" +
          "<strong>Prognosis:</strong><br>" +
          "Varies by severity; supportive therapies improve quality of life, but no cure exists.",      badge_ar: 'الأسبوع 8 - 24',
      title_ar: 'البورينسيفالي',
      description_ar: "هو تشوّه نادر في الدماغ حيث تتطور القشرة الدماغية بشكل غير طبيعي، مما يؤدي إلى طيّات صغيرة وغير منتظمة " +
             "بدلاً من البنية الطبيعية. يمكن أن يسبب هذا الحالة تأخّرًا في النمو، ونوبات صرع، وإعاقات عقلية، " +
             "ومشاكل حركية.<br>" +
             "<br>" +
             "<strong>الأسباب:</strong><br>" +
             "- <strong>طفرات جينية:</strong> (مثل TUBA1A، DCX).<br>" +
             "- <strong>العدوى أثناء الحمل:</strong> (مثل فيروس زيكا، CMV).<br>" +
             "- <strong>نقص الأكسجين:</strong> خلال تطور الجنين.<br>" +
             "- <strong>التعرض للسموم:</strong> أثناء الحمل.<br>" +
             "<br>" +
             "<strong>الأعراض:</strong><br>" +
             "- تأخر في النمو.<br>" +
             "- نوبات صرع.<br>" +
             "- ضعف أو تشنّج عضلي.<br>" +
             "- إعاقات معرفية ولغوية.<br>" +
             "<br>" +
             "<strong>التشخيص:</strong><br>" +
             "- <strong>الرنين المغناطيسي (MRI):</strong> لتصوير الدماغ.<br>" +
             "- <strong>الاختبارات الجينية:</strong> لتحديد الطفرات.<br>" +
             "<br>" +
             "<strong>العلاج:</strong><br>" +
             "- أدوية لعلاج نوبات الصرع.<br>" +
             "- علاج طبيعي، وعلاج نطقي، وعلاج وظيفي.<br>" +
             "- رعاية داعمة لصعوبات الحركة والتعلم.<br>" +
             "<br>" +
             "<strong>التكهن:</strong><br>" +
             "يختلف حسب الشدة؛ العلاجات الداعمة تحسّن جودة الحياة، لكن لا يوجد علاج شافٍ."


      },
      {
        img: '/img/Diseases/ventriculmegaly.gif',
        badge: '18th - 24th Weeks',
        title: 'Severe Ventriculomegaly',
        description: "Severe Ventriculomegaly is a condition where the brain's ventricles (fluid-filled spaces) become abnormally enlarged. " +
          "It is classified as severe when the ventricles measure ≥15 mm on ultrasound.<br>" +
          "<br>" +
          "<strong>Causes:</strong><br>" +
          "- <strong>Obstruction of CSF Flow:</strong> Conditions like hydrocephalus.<br>" +
          "- <strong>Brain Abnormalities:</strong> Polencephaly, neural tube defects, or brain malformations.<br>" +
          "- <strong>Infections:</strong> Congenital infections like Cytomegalovirus (CMV) or Toxoplasmosis.<br>" +
          "- <strong>Chromosomal Abnormalities:</strong> Linked to syndromes like Down or Edwards syndrome.<br>" +
          "<br>" +
          "<strong>Symptoms:</strong><br>" +
          "May not be apparent in utero, but postnatal symptoms can include:<br>" +
          "- Increased head size.<br>" +
          "- Developmental delays.<br>" +
          "- Neurological impairments.<br>" +
          "<br>" +
          "<strong>Diagnosis:</strong><br>" +
          "- <strong>Prenatal Ultrasound:</strong> Identifies ventricle size.<br>" +
          "- <strong>MRI:</strong> Provides detailed brain imaging.<br>" +
          "- <strong>Amniocentesis:</strong> To detect infections or genetic abnormalities.<br>" +
          "<br>" +
          "<strong>Treatment:</strong><br>" +
          "Depends on the cause.<br>" +
          "- <strong>Surgical Intervention:</strong> Ventriculoperitoneal (VP) shunt for hydrocephalus.<br>" +
          "- <strong>Supportive Therapies:</strong> Physical, occupational, and speech therapy.<br>" +
          "<br>" +
          "<strong>Prognosis:</strong><br>" +
          "Varies widely:<br>" +
          "- Mild cases may resolve spontaneously.<br>" +
          "- Severe cases depend on underlying causes and associated conditions.",      badge_ar: 'الأسبوع 18 - 24',
      title_ar: 'توسع البطينات الشديد',
      description_ar: "تضخم البطينات الدماغية الشديد (Severe Ventriculomegaly) هو حالة تصبح فيها بطينات الدماغ (المساحات المملوءة بالسائل) متضخمة بشكل غير طبيعي. " +
             "يُصنَّف بأنه شديد عندما تقيس البطينات ≥15 ملم في التصوير بالموجات فوق الصوتية.<br>" +
             "<br>" +
             "<strong>الأسباب:</strong><br>" +
             "- <strong>انسداد تدفق السائل الدماغي الشوكي (CSF):</strong> مثل حالات الاستسقاء الدماغي (hydrocephalus).<br>" +
             "- <strong>تشوهات الدماغ:</strong> مثل Polencephaly، عيوب الأنبوب العصبي، أو تشوهات الدماغ.<br>" +
             "- <strong>العدوى:</strong> العدوى الخلقية مثل الفيروس المضخم للخلايا (CMV) أو داء المقوسات (Toxoplasmosis).<br>" +
             "- <strong>الشذوذات الكروموسومية:</strong> مرتبطة بمتلازمات مثل متلازمة داون أو متلازمة إدواردز.<br>" +
             "<br>" +
             "<strong>الأعراض:</strong><br>" +
             "قد لا تظهر داخل الرحم، لكن الأعراض بعد الولادة يمكن أن تشمل:<br>" +
             "- زيادة في حجم الرأس.<br>" +
             "- تأخر في النمو.<br>" +
             "- اضطرابات عصبية.<br>" +
             "<br>" +
             "<strong>التشخيص:</strong><br>" +
             "- <strong>التصوير بالموجات فوق الصوتية قبل الولادة:</strong> لتحديد حجم البطينات.<br>" +
             "- <strong>الرنين المغناطيسي (MRI):</strong> لتقديم تصوير تفصيلي للدماغ.<br>" +
             "- <strong>البزل الأمنيوسي (Amniocentesis):</strong> لاكتشاف العدوى أو الشذوذات الجينية.<br>" +
             "<br>" +
             "<strong>العلاج:</strong><br>" +
             "يعتمد على السبب.<br>" +
             "- <strong>التدخل الجراحي:</strong> تركيب تحويلة بطينية بريتوانية (VP shunt) في حالات الاستسقاء الدماغي.<br>" +
             "- <strong>العلاجات الداعمة:</strong> العلاج الطبيعي، والعلاج الوظيفي، وعلاج النطق.<br>" +
             "<br>" +
             "<strong>التكهن:</strong><br>" +
             "يختلف بشكل واسع:<br>" +
             "- قد تُشفى الحالات الخفيفة تلقائيًا.<br>" +
             "- تعتمد الحالات الشديدة على الأسباب الكامنة والحالات المصاحبة."

      }
    ];  // Select the card list container
  const cardList = document.querySelector('.card-list');
  // Function to generate cards
  function generateCards() {
    // Clear existing cards first
    cardList.innerHTML = '';
    
    // Generate cards with proper language content
    diseases.forEach((disease, index) => {
      const cardItem = document.createElement('li');
      cardItem.classList.add('card-item', 'swiper-slide');
      // Use the current language setting to determine which content to show
      const useArabic = isArabic;
      cardItem.innerHTML = `
        <a class="card-link">
          <img src="${disease.img}" class="card-image">
          <p class="badge">${useArabic ? disease.badge_ar : disease.badge}</p>
          <h2 class="card-title">${useArabic ? disease.title_ar : disease.title}</h2>
          <p class="description">${useArabic ? disease.description_ar : disease.description}</p>
          <button class="card-btn" data-index="${index}">${useArabic ? 'اقرأ المزيد' : 'see more'}</button>
        </a>
      `;
      cardList.appendChild(cardItem);
    });
    
    // Reinitialize swiper after generating cards
    initSwiper();
  }

  // Generate initial cards
  generateCards();

  // Function to update the details section
  const detailsSection = document.querySelector('.Disease_detail_content');
  const diseaseImg = detailsSection.querySelector('.Disease_img img');
  const diseaseName = detailsSection.querySelector('.name span');
  const diseaseDescription = detailsSection.querySelector('.description span');
  // Function to display details of a selected disease
function displayDiseaseDetails(index) {
    const disease = diseases[index];
    diseaseImg.src = disease.img;
    diseaseName.textContent = isArabic ? disease.title_ar : disease.title;
    diseaseDescription.innerHTML = isArabic ? disease.description_ar : disease.description;
    
    // Track the current displayed disease index
    let currentIndexTracker = document.querySelector('.disease-detail-current-index');
    if (!currentIndexTracker) {
      currentIndexTracker = document.createElement('input');
      currentIndexTracker.type = 'hidden';
      currentIndexTracker.className = 'disease-detail-current-index';
      document.body.appendChild(currentIndexTracker);
    }
    currentIndexTracker.value = index;
  }// Initial load: display details of the first disease
  displayDiseaseDetails(0);
  // Apply saved language setting on page load
  // Use requestAnimationFrame for smoother transition
  requestAnimationFrame(() => {
    // Set initial language state based on localStorage
    if (isArabic) {
      document.documentElement.setAttribute('lang', 'ar');
      document.documentElement.setAttribute('dir', 'rtl');
      document.body.classList.add('rtl');
      langToggle.querySelector('.toggle-lang').textContent = 'العربية';
    } else {
      document.documentElement.setAttribute('lang', 'en');
      document.documentElement.setAttribute('dir', 'ltr');
      document.body.classList.remove('rtl');
      langToggle.querySelector('.toggle-lang').textContent = 'English';
    }
    
    // Update all content with the current language
    updateLanguage();
  });
  // Function to initialize card button listeners
  function initCardButtonListeners() {
    const cardButtons = document.querySelectorAll('.card-btn');
    cardButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const index = btn.getAttribute('data-index');
        displayDiseaseDetails(index);
        document.querySelector('#DiseaseDetails').scrollIntoView({ behavior: 'smooth' });
      });
    });
  }
  
  // Initialize card button listeners
  initCardButtonListeners();
});





// Dark mode toggle functionality
const darkModeToggle = document.getElementById('darkModeToggle');

darkModeToggle.addEventListener('click', () => {
  const isDarkMode = document.body.classList.contains('dark-mode');
  if (isDarkMode) {
    document.body.classList.remove('dark-mode');
    darkModeToggle.innerHTML='<span class="toggle-off"><i class="fa-regular fa-lightbulb"></i></span>';
    localStorage.setItem('darkMode', 'false');
  } else {
    document.body.classList.add('dark-mode');
    darkModeToggle.innerHTML='<span class="toggle-on"><i class="fa-solid fa-lightbulb"></i></span>';
    localStorage.setItem('darkMode', 'true');
  }
});

// Language toggle functionality
const langToggle = document.getElementById('langToggle');
let isArabic = false;

function updateLanguage() {
  const resourcesTitle = document.querySelector('.resources-title');
  const goToListBtn = document.querySelector('.btn a');
  const resourceDescriptions = document.querySelectorAll('.resource-description');
  const resourceLinks = document.querySelectorAll('.resource-link');
  
  // Add transition effect for smoother language change
  document.body.style.transition = 'opacity 0.3s ease';
  //document.body.style.opacity = '0.5';
  
  // Use requestAnimationFrame for smoother transitions
  requestAnimationFrame(() => {    if (isArabic) {
      // Switch to Arabic
      document.body.classList.add('rtl');
      document.documentElement.setAttribute('lang', 'ar');
      document.documentElement.setAttribute('dir', 'rtl');
      langToggle.querySelector('.toggle-lang').textContent = 'العربية';
      
      // Update other UI elements
      if (resourcesTitle) resourcesTitle.textContent = 'مصادر مفيدة';
      if (goToListBtn) goToListBtn.innerHTML = 'العودة للقائمة <i class="fa-solid fa-arrow-right"></i>';
      
      // Update resources section
      resourceDescriptions[0].textContent = 'نظرة شاملة عن البورينسيفالي من مؤسسة طب الأعصاب للأطفال.';
      resourceDescriptions[1].textContent = 'مقالة علمية تستكشف الاضطرابات ذات الصلة ودراسات الحالة.';
      resourceDescriptions[2].textContent = 'رؤى وأبحاث حول اضطرابات الجهاز العصبي النادرة.';
      resourceDescriptions[3].textContent = 'معلومات مفصلة عن كولفوسيفالي وعلاجها.';
      resourceDescriptions[4].textContent = 'مزيد من المقالات عن الاضطرابات العصبية والتنموية ذات الصلة.';
      resourceDescriptions[5].textContent = 'استكشاف مفصل للأبحاث الطبية المتعلقة بالأمراض النادرة.';
      resourceDescriptions[6].textContent = 'معلومات عن توسع البطينات، أسبابه، وخيارات العلاج.';
      
      // Update resource links to show Arabic translations
      resourceLinks.forEach((link, index) => {
        // Keep the href the same, just update the display text
        link.innerHTML = `موقع ${index + 1}`;
      });
      
      // Update currently displayed disease details
      const currentIndex = parseInt(document.querySelector('.card-btn[data-index]')?.getAttribute('data-index') || 0);
      displayDiseaseDetails(currentIndex);
        } else {
      // Switch to English
      document.body.classList.remove('rtl');
      document.documentElement.setAttribute('lang', 'en');
      document.documentElement.setAttribute('dir', 'ltr');
      langToggle.querySelector('.toggle-lang').textContent = 'English';
      
      // Update other UI elements
      if (resourcesTitle) resourcesTitle.textContent = 'Useful Resources';
      if (goToListBtn) goToListBtn.innerHTML = '<i class="fa-solid fa-arrow-left"></i>Go to List';
      
      // Update resources section
      resourceDescriptions[0].textContent = 'A comprehensive overview of porencephaly from the Child Neurology Foundation.';
      resourceDescriptions[1].textContent = 'A scientific article exploring related disorders and case studies.';
      resourceDescriptions[2].textContent = 'Insights and research about rare neurological disorders.';
      resourceDescriptions[3].textContent = 'Detailed information on colpocephaly and its management.';
      resourceDescriptions[4].textContent = 'Further articles on related neurological and developmental disorders.';
      resourceDescriptions[5].textContent = 'A detailed exploration of medical research related to rare diseases.';
      resourceDescriptions[6].textContent = 'Information on ventriculomegaly, its causes, and management options.';
      
      // Restore original link texts
      const originalLinkTexts = [
        'Child Neurology Foundation - Porencephaly',
        'NCBI Article - Detailed Study',
        'EurekaSelect - Neurological Insights',
        'Medlink - Colpocephaly',
        'EurekaSelect - Related Disorders',
        'EurekaSelect - Medical Research',
        'Nationwide Children\'s - Ventriculomegaly'
      ];
      
      resourceLinks.forEach((link, index) => {
        if (index < originalLinkTexts.length) {
          link.innerHTML = originalLinkTexts[index];
        }
      });
      
      // Update currently displayed disease details
      const currentIndex = parseInt(document.querySelector('.card-btn[data-index]')?.getAttribute('data-index') || 0);
      displayDiseaseDetails(currentIndex);
    }
      // Restore opacity after language change
    document.body.style.opacity = '1';
  });
  
  // Update the current displayed disease details if any
  const currentDetailIndex = document.querySelector('.disease-detail-current-index')?.value;
  if (currentDetailIndex) {
    displayDiseaseDetails(currentDetailIndex);
  }
}

langToggle.addEventListener('click', () => {
  isArabic = !isArabic;
  // Store language preference in localStorage
  localStorage.setItem('isArabic', isArabic);
  // Update language without page refresh
  updateLanguage();
  // Regenerate cards with the new language
  generateCards();
  // Reinitialize click event listeners for card buttons
  initCardButtonListeners();
});
